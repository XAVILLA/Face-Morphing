Full Project Report at: https://inst.eecs.berkeley.edu/~cs194-26/sp20/upload/files/proj3/cs194-26-act/
    
    
    
    To reproduce the result of this project
    To record the correspondence of an image: "python main.py {image path} record {given name for data}"
Then data of this image will be stored in the folder "points_data" as "{given name}_data.txt"

    To produce mean face of two images, use command
    "python main.py {image path 1} {image path 2} {given name 1} {given name 2} mean"
Then 3 frames will be stored in folder "{given name 1}_{given name 2}", with frame 0 and 2 being original images

     To produce morphing sequence of two images, use command
    "python main.py {image path 1} {image path 2} {given name 1} {given name 2} full"
Then 46 frames will be stored in folder "{given name 1}_{given name 2}", with frame 0 and 45 being original images

    To produce morphing gif from a folder with 46 frames, use command
    "python main.py {folderpath} gif"
Then the gif will be stored in the given folder

    To produce morphing video from a folder with 46 frames, use command
    "python main.py {folderpath} video"
Then the video will be stored in the given folder

    To produce the mean face of BAIR datas, use command
    "python main.py bair"
Result will be saved as "result.jpg"

    To wrap any face into another face's shape, use command
    "python main.py {imagepath1} {givename1} into {imagepath2} {givename2} {alpha}"
Where alpha denotes the alpha value used to make the wrapping.

    To wrap my face into BAIR, use command
    "python main.py images/zzx.jpg zzx into images/bair.jpg bair_mean 1"
This must be executed after producing the mean face of BAIR

    To perform caricatures and extrapolating, refer to "wrap any face into another face's shape", and use
an alpha value that's not within 0-1 range.

    To produce my face with female shape, use command
    "python main.py tofemale shape"
Result will be saved as "female_zzx_shape.jpg"

    To produce my face with female color, use command
    "python main.py tofemale color"
Result will be saved as "female_zzx_color.jpg"

    To produce my face with female shape and color, use command
    "python main.py tofemale both"
Result will be saved as "female_zzx_both.jpg"
