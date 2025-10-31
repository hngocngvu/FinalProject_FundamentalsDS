from dotenv import load_dotenv
import os 

# Simple validation
def get_api_base_dir():
    BASE_DIR = os.getcwd()
    load_dotenv(os.path.abspath(os.path.join(BASE_DIR,'..','.env')))
    API_KEY = os.getenv('API_KEY')
    return API_KEY, BASE_DIR

BASE_DIR = get_api_base_dir()[1]

if __name__ == "__main__":
    print(get_api_base_dir()[0])