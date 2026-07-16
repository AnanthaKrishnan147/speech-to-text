import os
from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError

try:
    # Initialize API 
    api = HfApi()
    user_info = api.whoami()
    
    print("--- LOGIN SUCCESSFUL ---")
    print(f"Logged in as: {user_info.get('name')}")
    # Using .get() prevents KeyError if the field is missing
    print(f"Email:        {user_info.get('email', 'Hidden or Not Provided')}")
    print(f"Token Type:   {user_info.get('auth', {}).get('accessToken', {}).get('role', 'N/A')}")

except LocalTokenNotFoundError:
    print("--- LOGIN FAILED ---")
    print("No token found on this machine. Run `huggingface-cli login` in your terminal.")
except Exception as e:
    print("--- ERROR OCCURRED ---")
    print(f"Could not authenticate. Details: {e}")
