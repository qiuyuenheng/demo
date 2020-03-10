from recognize.crnn_recognizer import PytorchOcr
recognizer = PytorchOcr()


def ch_recognize(image):
    text = recognizer.recognize(image)
    print(text)
    return text
