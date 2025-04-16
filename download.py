import requests

def download_ggfu_file():
    file_id = "1v5_eMAlNWBW34ahL2zkH9n8kH0K-UUI1"  # replace with your real ID
    url = f"https://drive.google.com/file/d/1v5_eMAlNWBW34ahL2zkH9n8kH0K-UUI1/view?usp=sharing"
    
    print("Downloading .ggfu file...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("models/hermes.ggfu", "wb") as f:
            f.write(response.content)
        print("File downloaded successfully!")
    else:
        print("Failed to download file.")

if __name__ == "__main__":
    download_ggfu_file()
