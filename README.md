# Syntactic and semantic dual enhanced bidirectional network for aspect sentiment triplet extraction

Installation

torch==1.7.1+cu110
numpy==1.21.6
thop==0.1.1
transformers==4.3.0
tqdm==4.65.0
allennlp==2.1.0
spacy==2.2.1
en-core-web-sm==2.2.0

Model Training

Replace lap14 for other datasets (eg 14res, 15res, 16res)

parser.add_argument("--dataset", default="lap14", type=str, choices=["lap14", "res14", "res15", "res16"],
                        help="specify the dataset")

Model Test

parser.add_argument('--mode', type=str, default="test", choices=["train", "test"], help='option: train, test')

Note

Although we have improved the indicators of the ASTE task to a certain extent, the model has high time complexity, and the effect of the part-of-speech-based pruning strategy is not obvious.

