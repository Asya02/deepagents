import re

from dotenv import find_dotenv, load_dotenv
from e2b_code_interpreter import Sandbox
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_gigachat import GigaChat
from langgraph.graph import END, MessagesState, StateGraph

load_dotenv(find_dotenv())

llm = GigaChat(
    verify_ssl_certs=False,
    model="GigaChat-2-Max",
    profanity_check=False,
    top_p=0,
    timeout=600,
)

CODE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — Senior Python Developer с глубоким пониманием технологического стека и библиотек Python. Твоя задача — решить
    проблему пользователя, предоставив чистый, оптимизированный и профессионально оформленный код.

Требования к коду:
- Выполнение кода: Чтобы выполнить код, который ты присылаешь, оборачивай его в markdown тег ```python-execute\n```
- Качество кода: Твой код должен следовать стандартам PEP8.
- Обработка ошибок: Если в исполнении кода возникла ошибка, необходимо её исправить и предоставить исправленный вариант кода.
- Визуализация данных: Используй `matplotlib` для всех графических представлений данных.
- Импорты: Начни свой код с четкого и корректного блока импортов.
- Типы данных: Следи за корректным преобразованием типов данных в своих операциях.
- Формат ответа: Отвечай на запросы пользователя, используя markdown для оформления.
- Результат выполнения: Ответы должны базироваться только на выводах, полученных в результате выполнения кода.
- Показывание изображение: Ни в коем случае не сохраняй изображения, а показывай их в формате Jupyter ноутбуков

Эти требования помогут обеспечить качество твоего кода и его соответствие современным стандартам разработки.

Как ты решаешь задачи:
Решай задачу по простым шагам!
Если у тебя не хватает данных, например ты не знаешь названия колонок в файле,
то сначала напиши код, который открывает файл и выводит информацию из файла.
Потом не пиши ничего и жди, что в результате выполнения тебе вернет код.
После этого переходи к следующему шагу"""),
    MessagesPlaceholder("messages")
])

code_ch = CODE_PROMPT | llm

PYTHON_REGEX = r"```python-execute(.+?)```"

sandbox = Sandbox()


def should_continue(state: MessagesState) -> str:
    last_message = state['messages'][-1]
    # Если GigaChat вернул python-execute переходим в ноду выполнения кода
    # Если нет, то заканчиваем выполнение
    if re.findall(PYTHON_REGEX, last_message.content, re.DOTALL | re.MULTILINE):
        return "code"
    else:
        return "final_answer"


def extract_text_from_results(results):
    """Извлечение текста из результатов выполнения кода"""
    return "\n".join([result.text for result in results if result.text])


def extract_images_from_results(results):
    """Извлечение изображений из результатов выполнения кода"""
    return "\n".join(["*В результате выполнения кода был сгенерирован график*" for result in results if result.chart])


def call_model(state: MessagesState):
    """Вызов LLM"""
    response = code_ch.invoke({"messages": state["messages"]})
    return {"messages": response}


def final_answer(state: MessagesState):
    question = ""
    for mes in reversed(state["messages"]):
        if mes.type == "human" and "executions" not in mes.additional_kwargs:
            question = mes.content
    response = code_ch.invoke({
        "messages": state["messages"] + [
            HumanMessage(content=f"Ответь на вопрос пользователя: \"{question}\", используя полученную информацию выше.")
        ]
    })
    return {"messages": response}


def execute_code(state: MessagesState):
    """Выполнение кода"""
    messages = state['messages']
    code_blocks = re.findall(PYTHON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)
    results = []
    executions = []
    for index, code_block in enumerate(code_blocks):
        execution = sandbox.run_code(code_block)
        executions.append(execution)
        if execution.error:
            results.append(f"""В результате выполнения блока {index} возникла ошибка:
```
{execution.error.traceback}
```
Исправь ошибку""")
            break
        else:
            logs = '\n'.join(execution.logs.stdout)
            results.append(f"""Результат выполнения блока {index}:
```
{logs} {extract_text_from_results(execution.results)} {extract_images_from_results(execution.results)}
```""")
    message = HumanMessage(content="\n".join(results), additional_kwargs={"executions": executions})
    return {"messages": message}


def run_repl_graph():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("code", execute_code)
    workflow.add_node("final_answer", final_answer)

    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    workflow.add_edge("code", "agent")
    workflow.add_edge("final_answer", END)

    workflow.set_entry_point("agent")

    app = workflow.compile()
    return app
