import glob
from os.path import basename

if __name__ == '__main__':
    file_list = glob.glob("dataset-ocr/*.jpg")
    output = [basename(a)+"\t" for a in file_list]
    with open("dataset-ocr/_ocr.tsv", "w") as f:
        f.write("\n".join(output))