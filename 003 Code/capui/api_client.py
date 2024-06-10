import webuiapi

# create API client
# api = webuiapi.WebUIApi()

# create API client with custom host, port
api = webuiapi.WebUIApi(host='127.0.0.1', port=7861)

# create API client with custom host, port and https
#api = webuiapi.WebUIApi(host='webui.example.com', port=443, use_https=True)

# create API client with default sampler, steps.
#api = webuiapi.WebUIApi(sampler='Euler a', steps=20)

# optionally set username, password when --api-auth=username:password is set on webui.
# username, password are not protected and can be derived easily if the communication channel is not encrypted.
# you can also pass username, password to the WebUIApi constructor.
api.set_auth('username', 'password')

def change_model(model):
    api.util_set_model(model)

def get_generate_progress():
    return api.get_progress()

def progress():
    return api.util_wait_for_ready()