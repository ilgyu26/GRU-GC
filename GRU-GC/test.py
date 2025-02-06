from models.gru_gc import GRU_GC
from options.options import GRUOptions, TrainingOptions
from preprocessing.preprocessing import data_preprocessing  # 데이터 로딩 추가
from util.util import plot

if __name__ == '__main__':
    gru_options = GRUOptions()
    gru_opt = gru_options.parse()

    training_options = TrainingOptions()
    training_opt = training_options.parse()

    # 데이터 로딩
    train_loader, val_loader = data_preprocessing(
        sequence_length=gru_opt.sequence_length, 
        num_shift=1,  # 기본값 유지
        batch_size=gru_opt.batch_size,
        shuffle=True
    )

    gru_gc = GRU_GC(gru_opt, training_opt)

    # 올바른 메서드 호출 방식 적용
    granger_matrix = gru_gc.nue(train_loader, val_loader)

    plot(granger_matrix)
