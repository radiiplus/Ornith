class ReportStore:
    def __init__(self, records):
        self.records = records

    def totals(self, tenant):
        return sum(self.records[tenant])
