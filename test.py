import tinker
service_client = tinker.ServiceClient()
print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)

from tinker_cookbook.hyperparam_utils import get_lr
model_name = "Qwen/Qwen3-4B-Instruct-2507"
recommended_lr = get_lr(model_name)
print(f"Recommended LR: {recommended_lr}")