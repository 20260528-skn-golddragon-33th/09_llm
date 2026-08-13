"""
LangChain 기본 체험 - "3단계를 묶는다"를 직접 눈으로 확인하기

실행:  python 0_practice_langchain_basic.py
필요:  pip install langchain-openai langchain-core python-dotenv
       그리고 09_llm/.env 안에 OPENAI_API_KEY=... 가 있어야 함
"""

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True))  # .env에서 OPENAI_API_KEY 로드

# ── LangChain의 3개 블록 ─────────────────────────────
from langchain_openai import ChatOpenAI                       # ② 모델
from langchain_core.prompts import ChatPromptTemplate         # ① 프롬프트
from langchain_core.output_parsers import StrOutputParser     # ③ 파서

# ① 프롬프트: {topic} 자리를 나중에 채움
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 한 문장으로 짧게 답하는 챗봇이다."),
    ("human",  "{topic}에 대해 알려줘"),
])

# ② 모델
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ③ 파서: 응답 객체에서 텍스트만 꺼냄
parser = StrOutputParser()

# ★ 핵심: 3개를 파이프(|)로 "묶는다"
chain = prompt | model | parser


print("=" * 50)
print("[1] invoke() — 하나 실행")
print("=" * 50)
answer = chain.invoke({"topic": "파이썬"})
print(answer)


print("\n" + "=" * 50)
print("[2] batch() — 여러 개 한꺼번에")
print("=" * 50)
answers = chain.batch([
    {"topic": "파이썬"},
    {"topic": "자바"},
    {"topic": "러스트"},
])
for a in answers:
    print("-", a)


print("\n" + "=" * 50)
print("[3] stream() — 조금씩 흘러나오게")
print("=" * 50)
for chunk in chain.stream({"topic": "랭체인"}):
    print(chunk, end="", flush=True)
print()
