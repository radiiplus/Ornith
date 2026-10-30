use std::env;
use std::fs::File;
use std::io::{self, Read};

const CHUNK_BYTES: usize = 800;
const OVERLAP_BYTES: usize = 100;
const MAX_RESULTS: usize = 6;

struct Candidate {
    ordinal: usize,
    score: usize,
    text: String,
}

fn scan<R: Read>(mut reader: R, source: &str, query: &str, max_bytes: usize) -> io::Result<String> {
    let terms: Vec<String> = query
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|term| !term.is_empty())
        .take(12)
        .map(str::to_ascii_lowercase)
        .collect();
    if terms.is_empty() || max_bytes == 0 {
        return Ok(String::new());
    }

    let mut carry = Vec::with_capacity(OVERLAP_BYTES);
    let mut scratch = [0_u8; CHUNK_BYTES - OVERLAP_BYTES];
    let mut best: Vec<Candidate> = Vec::with_capacity(MAX_RESULTS);
    let mut ordinal = 0;
    loop {
        let mut bytes = carry.clone();
        let mut added = 0;
        while bytes.len() < CHUNK_BYTES {
            let wanted = (CHUNK_BYTES - bytes.len()).min(scratch.len());
            let count = reader.read(&mut scratch[..wanted])?;
            if count == 0 {
                break;
            }
            bytes.extend_from_slice(&scratch[..count]);
            added += count;
        }
        if added == 0 {
            break;
        }

        let text = String::from_utf8_lossy(&bytes).into_owned();
        let lower = text.to_ascii_lowercase();
        let score = terms.iter().map(|term| lower.matches(term).count()).sum();
        if score > 0 {
            best.push(Candidate {
                ordinal,
                score,
                text,
            });
            best.sort_by(|a, b| b.score.cmp(&a.score).then(a.ordinal.cmp(&b.ordinal)));
            best.truncate(MAX_RESULTS);
        }
        carry = bytes[bytes.len().saturating_sub(OVERLAP_BYTES)..].to_vec();
        ordinal += 1;
    }

    let mut output = String::new();
    for candidate in best {
        let section = format!(
            "[SOURCE {source} chunk {}]\n{}",
            candidate.ordinal, candidate.text
        );
        let separator = if output.is_empty() { "" } else { "\n\n" };
        if output.len() + separator.len() + section.len() <= max_bytes {
            output.push_str(separator);
            output.push_str(&section);
        }
    }
    Ok(output)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: rust-context-probe FILE QUERY MAX_BYTES");
        std::process::exit(2);
    }
    let max_bytes: usize = args[3]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "MAX_BYTES must be an integer"))?;
    if max_bytes > 8192 || File::open(&args[1])?.metadata()?.len() > 128 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "input limit exceeded",
        ));
    }
    print!(
        "{}",
        scan(File::open(&args[1])?, &args[1], &args[2], max_bytes)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::scan;

    #[test]
    fn finds_late_marker_with_bounded_output() {
        let corpus = format!("{} unique_marker result", "alpha ".repeat(1000));
        let output = scan(corpus.as_bytes(), "notes.txt", "unique_marker", 1000).unwrap();
        assert!(output.contains("unique_marker"));
        assert!(output.contains("[SOURCE notes.txt chunk "));
        assert!(output.len() <= 1000);
    }

    #[test]
    fn empty_query_selects_nothing() {
        assert_eq!(scan("alpha".as_bytes(), "notes", "!", 1000).unwrap(), "");
    }
}
