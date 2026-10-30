def allowed(user, tenant):
    return tenant in user['tenants']
