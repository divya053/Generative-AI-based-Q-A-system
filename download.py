import requests

def download_ggfu_file():
    file_id = "1A2B3C4D5E6F7G8H"  # replace with your real ID
    url = f"https://drive.google.com/drive/folders/1K8qumxMXVZJ0P67nbDxJzqh-xTpxiCil?usp=drive_link"
    
    print("Downloading .ggfu file...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("data/file.ggfu", "wb") as f:
            f.write(response.content)
        print("File downloaded successfully!")
    else:
        print("Failed to download file.")

if __name__ == "__main__":
    download_ggfu_file()
