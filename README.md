# Общие сведения
Тренировка выполнялась на датасете размером 300k твитов. Пробовал полный сет, но работало очень долго.

График с метриками:

![rouge](models/plots/run_001_loss_rouge.png)

![ppl](models/plots/run_001_loss_ppl.png)

Из графиков можно видеть, что LSTM проделала основной путь в обучении (для 300k записей)

# Avg. ROUGE-L

|LSTM Greedy|Transformer Greedy|LSTM Sampling|Transformer Sampling|
|:-:|:-:|:-:|:-:|
|0.070653|0.051920|0.071244|0.049679|
|Запись|Запись|Запись|Запись|

Обученная модель LSTM лучше справляется задачей как на жадной генерации, так и при семплировании. Если посмотреть на сами сгенерованные тексты, то и тут видно, как трансформер может добавлять неуместные переносы строк. В целом качество предсказания для обеих моделей оставляет желать лучшего. 

# LSTM жадная генерация
PROMT: - awww, that's a bummer. you shoulda g
PREDICTION: - awww, that's a bummer. you shoulda gd you!!!!!!!!!!!!!!!!!!

PROMT: is upset that he can't update his facebook by texting 
PREDICTION: is upset that he can't update his facebook by texting !!!!!!!!!!!!!!!!!!!!!!

PROMT: i dived many times for the ball. manage
PREDICTION: i dived many times for the ball. manage to get a new one.!!!!!!!!!!!!!!

PROMT: my whole body feels itc
PREDICTION: my whole body feels itc!!!!!!!!!!!!!!!!!!!!

PROMT: no, it's not behaving at all. i'm mad. why am 
PREDICTION: no, it's not behaving at all. i'm mad. why am ive been up since 4am and i'm so tired!!!!!!!!!


# LSTM с семплированием
PROMT: - awww, that's a bummer. you shoulda g
PREDICTION: - awww, that's a bummer. you shoulda give emic was great.!!!! makes me cry!!!!!!.

PROMT: is upset that he can't update his facebook by texting 
PREDICTION: is upset that he can't update his facebook by texting iphone!!!! no more red.!!!!!!!!!!

PROMT: i dived many times for the ball. manage
PREDICTION: i dived many times for the ball. manage to do some more than the other work!! my mouth is killing me.!!x.

PROMT: my whole body feels itc
PREDICTION: my whole body feels itc! and my best friends and im almost over!!!!!!!!!!!

PROMT: no, it's not behaving at all. i'm mad. why am 
PREDICTION: no, it's not behaving at all. i'm mad. why am ??????!!! i can't see you tomorrow..! &lt;3!&lt



# TRANSFORMER с семплированием
PROMT: - awww, that's a bummer. you shoulda g
PREDICTION: ee at the idea of a c-section. so you can get a feel for how the c

PROMT: is upset that he can't update his facebook by texting i
PREDICTION: Message














...

PROMT: no, it's not behaving at all. i'm mad. why am 
PREDICTION: ive seen my wife being harassed by a man who I like to call "em, I'm mad

# TRANSFORMER жадная генерация
PROMT: - awww, that's a bummer. you shoulda g
PREDICTION: osh, but I'm not going to be able to do that. I'm going to be able

PROMT: is upset that he can't update his facebook by texting i
PREDICTION: Message.


















...


I'm not sure if it's a good idea to just say

Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
