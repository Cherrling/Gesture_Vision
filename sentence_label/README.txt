1. How to process the compressed data in the folder[sentence]?
a. download the splited files of it.
    1- download csl-daily-frames-512x512.tar.gz_00 ---- csl-daily-frames-512x512.tar.gz_09 
    2- cat these files on Ubuntu.
        cat csl-daily-frames-512x512.tar.gz_* >  csl-daily-frames-512x512.tar.gz
c. extract the file
    tar -xvzf csl-daily-frames-512x512.tar.gz

2. How to read the annotation file?
import pickle
with open('csl2020ct_v2.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
print(data['info'][0])

3. What's the meaning of keys in the above file?
info: all the annotations
    name: video name
    length: the number of frames in the video
    label_gloss: gloss sequence stored in [List] for sign language recognition
    label_char: char sequence stored in [List] for sign language translation. In our experiments, we use this as SLT target.
    label_word: word sequence stored in [List] for sign language translation. Just for reference.
    signer: the id of some signer. It starts from 0.
    time: how many times the same signer performs the same sentence. It starts from 0. Actually, no signer performs the same sentence twice.
gloss_map: vocabulary for glosses
char_map: vocabulary for chars
word_map: vocabulary for words

4. Where is the split of train/dev/test?
In 'split_1.txt'

5. How to understand the meanings of Chinese sentences?
for example: 
    import pickle
    with open('csl2020ct_v1.pkl', 'rb') as f:
        data = pickle.load(f)
    print(''.join(data['info'][1000]['label_char']))
You will see '下个星期一上班。'
Then you can translation it with google. We will consider adding English translations in the future.
