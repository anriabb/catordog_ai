For Debian-Based operating system 
   cd ~project
1. sudo apt install python3-venv
2. python3 -m venv venv
3. source venv/bin/activate
4. pip install tensorflow keras matplotlib numpy
5. cd ~project
6. python main.py
7. Database of images: https://www.kaggle.com/datasets/chetankv/dogs-cats-images?resource=download
   put the images like this:
   ...other files in project
   data/
        train/
            cats/ (put there train-cats images, ~4000 images)
            dogs/ (put there train-dogs images, ~4000 images)
        val/
            cats/ (put there test-cats images, ~1000 images)
            dogs/ (put there test-dogs images, ~1000 images)
