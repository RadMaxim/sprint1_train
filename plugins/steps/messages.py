from airflow.providers.telegram.hooks.telegram import TelegramHook # импортируем хук телеграма

def send_telegram_success_message(context): # на вход принимаем словарь с контекстными переменными
    hook = TelegramHook(token='8823691386:AAEF4pvMLBefQiI0PDuLGJhPTULKfUVBq4w', chat_id='-5144673049')
    dag = context['dag'].dag_id
    run_id = context['run_id']
    
    message = f'Исполнение DAG {dag} с id={run_id} прошло успешно!' # определение текста сообщения
    hook.send_message({
        'chat_id': '-5144673049',
        'text': message
    }) # отправление сообщения