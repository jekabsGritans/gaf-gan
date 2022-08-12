from dataset import ForexData

def get_preprocessed(seq_length):
    transforms = [

    ]
    dataset = ForexData(seq_length)
    return dataset


if __name__ == "__main__":
    dataset = ForexData(32)
    import numpy as np
    import matplotlib.pyplot as plt
    
    sample = dataset[0].numpy()

    plt.plot(sample)
    plt.show()
