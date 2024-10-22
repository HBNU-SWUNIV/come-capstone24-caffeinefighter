from googletrans import Translator
translator = Translator()

txt = '''이미 일어나 버린 일에 대해 뒤늦게 이유를 늘어놓아 봐야 사실은 아무것도 변하지 않는다. 그런데 사람들은 왜 동기다, 경위다, 이유다 하는 것을 요구하는 것일까.'''
res = translator.translate(txt, src='ko', dest='en')

res.text

print(res)