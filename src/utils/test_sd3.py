import requests


imgage = requests.post("http://localhost:5008/generate_image?dog")
    
    image.save('test.png')
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.show()

