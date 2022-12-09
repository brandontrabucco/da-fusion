from semantic_aug.datasets.spurge import SpurgeDataset


if __name__ == "__main__":

    dataset = SpurgeDataset("train", 5)

    print("finished", len(dataset))