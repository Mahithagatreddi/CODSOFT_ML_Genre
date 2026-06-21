import os

def generate_mock_data(filepath="data/train_data.txt"):
    """
    Generates a small mock dataset in the Kaggle format to test the pipeline.
    Format: ID ::: TITLE ::: GENRE ::: DESCRIPTION
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    mock_data = [
        "1 ::: Spaceship Alpha (2050) ::: sci-fi ::: An astronaut travels through a black hole to save humanity from alien invasion.",
        "2 ::: The Funny Guy (2020) ::: comedy ::: A hilarious look at a stand-up comedian who accidentally becomes mayor.",
        "3 ::: Mars Attack (1998) ::: sci-fi ::: Aliens from Mars attack Earth using advanced laser technology and flying saucers.",
        "4 ::: Laugh Out Loud (2015) ::: comedy ::: Two friends go on a road trip and find themselves in the funniest situations ever.",
        "5 ::: Galactic War (2200) ::: sci-fi ::: The final battle for the galaxy begins when the dark empire strikes the rebel base.",
        "6 ::: The Silly Dog (2010) ::: comedy ::: A family adopts a golden retriever that can somehow talk and tell jokes.",
        "7 ::: Space Explorers (2030) ::: sci-fi ::: A team of scientists build a warp drive to explore distant galaxies and planets.",
        "8 ::: Crazy Neighbors (2005) ::: comedy ::: Moving to the suburbs turns into a nightmare when the neighbors are completely crazy.",
        "9 ::: Alien Resurrection (2100) ::: sci-fi ::: The alien queen returns to exact revenge on the space colony.",
        "10 ::: Pranksters (2018) ::: comedy ::: College students try to pull off the biggest prank in university history."
    ]
    
    with open(filepath, "w", encoding="utf-8") as f:
        for line in mock_data:
            f.write(line + "\n")
            
    print(f"Mock data generated at {filepath}")

if __name__ == "__main__":
    generate_mock_data()
