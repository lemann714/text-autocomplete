# Состав репозитория

- data_utils.py - обработка данных
- lstm_model.py - LSTM модель
- train_lstm.py - обучение и оценка LSTM
- inference_lstm.py - автогенерация текста с LSTM
- inference_transformer.py -  автогенерация текста с трансформером
- common.py - общие функции
- compare.py - сравнение моделей
- solution.ipynb - jupyter notebook в котром весь код
- models - в директории находится чекпоинт с лучшими весами, а также полная лучшая модель (после обучения на VM с GPU)

# Как пользоваться репозиторием

1. Клонируем

2. Создаем окружение:

`python3 -m venv .venv`
 
3. Активируем:

`source .venv/bin/activate`

4. Устанавливаем зависимости: 

`pip install -r requirements.txt`

## Сравнение моделей

Переходим в ./src

Запускаем сравнение:

`python3 compare.py`

## Запуск обучения

Чтобы запустить обучение, запустите модуль train_lstm.py:

Запускаем обучение:

`python3 train_lstm.py`

Если у вас определится cuda, то обучение будет на всем датасете, иначе - на выборке из 50 элементов (для дебага на локальной тачке). Артефакты обучения
будут сохранены в **models**. 

Внимание: Если запустить обучение, то мои чекпоинты пересоздадутся!

# Вычислительный эксперимент

Тренировка выполнялась на датасете размером 300k твитов. Пробовал полный сет, но работало очень долго.

Метрики:

![rouge](models/plots/run_001_loss_rouge.png)

![ppl](models/plots/run_001_loss_ppl.png)

# Avg. ROUGE-L

|LSTM Greedy|Transformer Greedy|LSTM Sampling|Transformer Sampling|
|:-:|:-:|:-:|:-:|
|0.070653|0.051920|0.071244|0.049679|

# Примеры автогенерации текста

| Модель       | Тип генерации     | PROMT                                          | PREDICTION                                                                                                                                           |
|--------------|-------------------|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| LSTM         | Семплирование     | - awww, that's a bummer. you shoulda g        | - awww, that's a bummer. you shoulda gadmfao.!! and do my hard work!!!!!!                                                                         |
|              |                   | is upset that he can't update his facebook by texting | is upset that he can't update his facebook by texting iphone!!!!                                                                                    |
|              |                   | i dived many times for the ball. manage        | i dived many times for the ball. manage. i want to see you!!!!! ugh!!!!                                                                            |
|              |                   | my whole body feels itc                        | my whole body feels itcetts. no one wants to help it.!!!!                                                                                           |
|              |                   | no, it's not behaving at all. i'm mad. why am  | no, it's not behaving at all. i'm mad. why am ??????????????????????????????????????????????????! and i don't want to hear that! you have been           |
|              |                   | not the w                                      | not the wierd i should be away!! its raining!!! lol!!!!!!                                                            |
|              |                   | need                                           | need iphone?!! that is what i get to be done today! i am so tired and                                                |
|              |                   | hey long t                                     | hey long tuesday. have to say. i think i have a headache..!! for a few days.                                          |
|              |                   | nope they di                                   | nope they dius bought an issue!!!!!!!! they are probably stupid.!!!                                                     |
|              |                   | que me                                         | que me !!! wish i had the suncream at this place!!!!!!!!!s                                                             |
|              |                   | spring break in plain                          | spring break in plain weather just not looking forward to being sick!!!!!!!!!!!!!!                                  |
|              |                   | i just re-pi                                   | i just re-pi. please please do i need a hug but im still not sure if i can get into the one                            |
|              |                   | i couldn't bear to watch it. and i thoug      | i couldn't bear to watch it. and i thoug i missed it!!! i should take a nap.!! our food has arrived!!                  |
|              |                   | it it counts, idk why i did eith              | it it counts, idk why i did eith!!! all my friends are with him.! i know and we're lucky..! that                       |
|              |                   | i would've been the first, but i didn't have a gun.| i would've been the first, but i didn't have a gun. but i have no idea what i have.!! :/!!!! i miss the                 |
|              |                   | i wish i got to watch it with you!!           | i wish i got to watch it with you!! ive been kinda disappointed about you, i'm really gonna miss u!!! and it would be |
|              |                   | hollis' death scene will hurt me severely to w | hollis' death scene will hurt me severely to wled the lottery!!!!!!!!!!! not. but it's too                             |
|              |                   | about to                                       | about to ive got an extra hr ago! bah boo, now i can't breathe for the night!                                        |
|              |                   | ahh ive always wanted to se                    | ahh ive always wanted to seperate uk online.! i can't go to bamboozle will be the same without                      |
|              |                   | oh dear. were you drinking out                 | oh dear. were you drinking out for a while? i think i'm going to fail the poor tavids.! will be                         |
| TRANSFORMER  | Семплирование     | - awww, that's a bummer. you shoulda g        | azillionaire, and btw, I'm a huge fan of him, but I don't                                                          |
|              |                   | is upset that he can't update his facebook by texting i | 'll see if he can do that.‍                                                                |
|              |                   | i dived many times for the ball. manage        |  and score an incredible goal, and it is always great to see how he got his hands on the                                |
|              |                   | my whole body feels itc                        | ally, and this is all good for the rest of you, too."                                                               |
|              |                   | no, it's not behaving at all. i'm mad. why am  | ike. i'm mad. i'm mad. i'm mad. i'm mad. i'm mad. i'm                                                                  |
|              |                   | not the w                                      | isp and other parts, you'll notice that the two different parts are very similar (one on one                          |
|              |                   | need                                           | __________ __________ __________ __________ __________ __________ __________                                        |
|              |                   | hey long time no see! yes.. rains a bit ,on    |  his way to the town.                                                                                                  |
|              |                   | nope they di                                   | em.                                                                                                                   |
|              |                   | que me                                         | vernacular and all of it is a little bit different. I've never really heard the                                       |
|              |                   | spring break in plain                          |  sight, but it is also clear that there are other reasons why this isn't the case.                                     |
|              |                   | i just re-pi                                   | ased the last game in the game. This is the last game in                                                              |
|              |                   | i couldn't bear to watch it. and i thoug      |  got a new job as a reporter for the Associated Press. it was a wonderful job, the only                               |
|              |                   | it it counts, idk why i did eith              |  one.                                                                                                                 |
|              |                   | i would've been the first, but i didn't have a gun.|  I can't see why i was the first. I can't see why i was the first.                                                  |
|              |                   | i wish i got to watch it with you!!           | !!!! I didn't see this until last year, I didn't see this until last year,                                            |
|              |                   | hollis' death scene will hurt me severely to w | ade through the mud. I have to put my hand down to give it a bit more of                                             |
|              |                   | about to                                       | irclenet.                                                                                                             |
|              |                   | ahh ive always wanted to se                    | gwit.                                                                                                                 |
|              |                   | oh dear. were you drinking out                 |  of the toilet? I was drinking out of the toilet. I was drinking out of the toilet                                    |
| LSTM         | Жадная генерация  | - awww, that's a bummer. you shoulda g        | - awww, that's a bummer. you shoulda gd you!!!!!!!!!!!!!!!!!!                                                       |
|              |                   | is upset that he can't update his facebook by texting | is upset that he can't update his facebook by texting !!!!!!!!!!!!!!!!!!!!!!                                         |
|              |                   | i dived many times for the ball. manage        | i dived many times for the ball. manage to get a new one.!!!!!!!!!!!!!!                                             |
|              |                   | my whole body feels itc                        | my whole body feels itc!!!!!!!!!!!!!!!!!!!!!                                                                           |
|              |                   | no, it's not behaving at all. i'm mad. why am  | no, it's not behaving at all. i'm mad. why am ive been up since 4am and i'm so tired!!!!!!!!!                       |
|              |                   | not the w                                      | not the wknds!!!!!!!!!!!!!!!!!                                                                                         |
|              |                   | need                                           | need iphone!!!!!!!!!!!!!!!!!!                                                                                          |
|              |                   | hey long t                                     | hey long tuesday! i'm so sorry to hear that!!!!!!!!!!!                                                               |
|              |                   | nope they di                                   | nope they diy is a good idea.!!!!!!!!!!!!!!                                                                           |
|              |                   | que me                                         | que me !!!!! i'm so sorry to hear that!!!!!!!!!!                                                                      |
|              |                   | spring break in plain                          | spring break in plain car is not working!!!!!!!!!!!!!!                                                                |
|              |                   | i just re-pi                                   | i just re-pi-in-law.!! i'm not going to be able to go to the beach.                                                   |
|              |                   | i couldn't bear to watch it. and i thoug      | i couldn't bear to watch it. and i thoug. i'm so sorry.!! i'm not sure if i can get it.!!                           |
|              |                   | it it counts, idk why i did eith              | it it counts, idk why i did eith to do it!!!!!!!!!!!!!!!!!                                                           |
|              |                   | i would've been the first, but i didn't have a gun.| i would've been the first, but i didn't have a gun.!!!!!!!!!!!!!!!!!!!!!                                             |
|              |                   | i wish i got to watch it with you!!           | i wish i got to watch it with you!! ive been so busy with my friends!!!!!!!!!!!!                                     |
|              |                   | hollis' death scene will hurt me severely to w | hollis' death scene will hurt me severely to wana go to the beach!!!!!!!!!!!!!!!                                     |
|              |                   | about to                                       | about to ive a lot of work to do!!!!!!!!!!!!!                                                                         |
|              |                   | ahh ive always wanted to se                    | ahh ive always wanted to seattle to go to the beach!!!!!!!!!!!!!!                                                   |
|              |                   | oh dear. were you drinking out                 | oh dear. were you drinking out?!! i'm not sure if i can't find it!!!!!!!                                            |
| TRANSFORMER  | Жадная генерация  | - awww, that's a bummer. you shoulda g        | osh, but I'm not going to be able to do that. I'm going to be able                                                   |
|              |                   | is upset that he can't update his facebook by texting i | Message.                                                                                                              |
|              |                   | i dived many times for the ball. manage        |  to get the ball out of the box.                                                                                      |
|              |                   | my whole body feels itc                        | oughing.”                                                                                                             |
|              |                   | no, it's not behaving at all. i'm mad. why am  | ive seen this? I'm not sure if it's a good idea to just say                                                          |
|              |                   | not the w                                      | isest thing to do.”                                                                                                   |
|              |                   | need                                           | __________________                                                                                                    |
|              |                   | hey long time no see! yes.. rains a bit ,on    |  the way to the beach, but I'm not sure if I'm going to get to the beach                                             |
|              |                   | nope they di                                   | atribe the same thing.”                                                                                               |
|              |                   | que me                                         | !!!!                                                                                                                  |
|              |                   | spring break in plain                          |  sight.                                                                                                               |
|              |                   | i just re-pi                                   | pped the $1.5 million mark.                                                                                          |
|              |                   | i couldn't bear to watch it. and i thoug      | re a man who is not a man.                                                                                            |
|              |                   | it it counts, idk why i did eith              | .                                                                                                                     |
|              |                   | i would've been the first, but i didn't have a gun.|                                                                                                                        |
|              |                   | i wish i got to watch it with you!!           | __________________                                                                                                    |
|              |                   | hollis' death scene will hurt me severely to w | ager that the police will not be able to arrest him. The police have been                                            |
|              |                   | about to                                       |                                                                                                                       |
|              |                   | ahh ive always wanted to se                    | perate from the other side of the world.                                                                              |
|              |                   | oh dear. were you drinking out                 |  of the bottle?                                                                                                       |

# Что лучше использовать

Судя по метрикам, обученная модель LSTM лучше справляется задачей как на жадной генерации, так и при семплировании. Если посмотреть на сами сгенерованные тексты, то трансформер может выдавать такую девиацию как множество последовательных переносов строк, что совсем не характерно для твитов. В целом, обе модели оставляют желать лучшего. Чтобы эксперимент был честным, стоило бы дообучить трансформер

Вывод: для автогенерации твитов текущая обученная LSTM работает немного лучше чем трансформер distilbert (без дообучения). Видимо, трансформер был обучен на совершенно других текстах.


# P.S.

- Обучение на всем датасете почему-то шло очень долго. Скорее всего я что-то сделал не так и такие приемы как .to(device) и pin_memory=True были недостаточны для организации нормальной работы обучающего цикла

- Когда кода было мало, все было нормально. Но как только его становилось больше, стало сложнее дебажить и управлять переменными и константами, поэтому код плохой

- Плагин для ssh это какой-то отстой. Думаю те, кто первый раз слышат про ssh и встрянут на этапе подключения через vscode сильно испортят себе нервы. Поэтому подключился по старинке. Vim - решает
