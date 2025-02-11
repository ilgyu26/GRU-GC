from models.gru_gc import GRU_GC
from data.data_preprocessing import preprocess_data
from util.util import plot

if __name__ == '__main__':
    dataset = preprocess_data(sequence_length=20)
    gru_gc = GRU_GC()
    granger_matrix = gru_gc.gru_gc(dataset)
    plot(granger_matrix)