# BERTopic 토픽 모델링

특허 문서 분석을 위한 BERTopic 기반 토픽 모델링 도구입니다.

## 주요 기능

- 특허 문서에 특화된 전처리 (불용어 제거, 형식어 제거 등)
- OpenAI 임베딩을 활용한 고품질 문서 벡터화
- Grid Search를 통한 최적 파라미터 자동 탐색
- 토픽 품질 평가 지표 제공
- 간편한 사용을 위한 모듈화된 구조

## 설치

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드 (최초 1회)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

## 사용 방법

### 1. 기본 사용

```bash
# 환경변수로 API 키 설정
export OPENAI_API_KEY="your-api-key"

# 토픽 모델링 실행
python main.py --input data.xlsx --output results/
```

### 2. Grid Search 포함

```bash
python main.py \
    --input data.xlsx \
    --output results/ \
    --grid-search
```

### 3. 임베딩 재사용

```bash
# 임베딩 생성 및 저장
python main.py \
    --input data.xlsx \
    --output results/ \
    --save-embeddings embeddings.npy

# 저장된 임베딩 로드
python main.py \
    --input data.xlsx \
    --output results/ \
    --load-embeddings embeddings.npy
```

### 4. Python 코드로 사용

```python
import pandas as pd
from src.preprocessing import TextPreprocessor
from src.topic_model import BERTopicModel
from src.metrics import TopicMetrics
from src.config import Config

# API 키 설정
Config.OPENAI_API_KEY = "your-api-key"

# 데이터 로드
df = pd.read_excel("data.xlsx")
documents = df['번역'].tolist()

# 전처리
preprocessor = TextPreprocessor()
processed_docs = preprocessor.preprocess(documents)

# 모델 생성 및 임베딩
model = BERTopicModel(
    openai_api_key=Config.OPENAI_API_KEY
)
embeddings = model.create_embeddings(processed_docs)

# Grid Search로 최적 파라미터 찾기
best_params, grid_results = model.grid_search(
    documents=processed_docs,
    embeddings=embeddings
)

# 모델 학습
topic_model = model.fit(
    documents=processed_docs,
    embeddings=embeddings,
    umap_params=best_params['umap_params'],
    hdbscan_params={
        'min_cluster_size': best_params['min_cluster_size'],
        'min_samples': best_params['min_samples'],
        'cluster_selection_epsilon': best_params['epsilon'],
        'metric': 'euclidean',
        'prediction_data': False
    }
)

# 평가 지표 확인
topic_info = topic_model.get_topic_info()
metrics = TopicMetrics.calculate_metrics(topic_info)
TopicMetrics.print_metrics(metrics)

# 결과 저장
topic_info.to_excel("topic_info.xlsx")
model.save_model("bertopic_model")
```

## 프로젝트 구조

```
.
├── src/
│   ├── __init__.py          # 패키지 초기화
│   ├── preprocessing.py     # 텍스트 전처리
│   ├── topic_model.py       # BERTopic 모델링
│   ├── metrics.py           # 평가 지표
│   └── config.py            # 설정 관리
├── main.py                  # 메인 실행 스크립트
├── requirements.txt         # 패키지 의존성
└── README.md               # 이 파일
```

## 모듈 설명

### 1. preprocessing.py

특허 문서 전처리를 담당합니다.

- **TextPreprocessor**: 텍스트 전처리 클래스
  - `normalize_text()`: 텍스트 정규화
  - `remove_patent_elements()`: 특허 형식어 제거
  - `tokenize_and_filter()`: 토큰화 및 불용어 제거
  - `preprocess()`: 전체 전처리 파이프라인

### 2. topic_model.py

BERTopic 모델링을 담당합니다.

- **BERTopicModel**: BERTopic 모델 래퍼
  - `create_embeddings()`: 문서 임베딩 생성
  - `grid_search()`: 최적 파라미터 탐색
  - `fit()`: 모델 학습
  - `save_model()` / `load_model()`: 모델 저장/로드

### 3. metrics.py

토픽 모델 평가 지표를 제공합니다.

- **TopicMetrics**: 평가 지표 클래스
  - `calculate_metrics()`: 모든 평가 지표 계산
  - `calculate_score()`: 종합 점수 계산
  - `print_metrics()`: 평가 지표 출력

**평가 지표:**
- `outlier_ratio`: 노이즈 비율 (낮을수록 좋음)
- `n_topics_clean`: 유효 토픽 개수
- `h_norm`: 정규화 엔트로피 (높을수록 균등 분포)
- `top1_clean`: 최대 토픽 비율 (너무 크면 불균형)
- `gini_clean`: 지니 계수 (낮을수록 균등 분포)

### 4. config.py

설정을 관리합니다.

- **Config**: 설정 클래스
  - OpenAI API 설정
  - BERTopic 파라미터
  - Grid Search 설정
  - 전처리 설정

## 명령행 인자

```
--input TEXT          입력 엑셀 파일 경로 (필수)
--output TEXT         출력 디렉토리 경로 (기본: ./results)
--text-column TEXT    분석할 텍스트 컬럼명 (기본: 번역)
--api-key TEXT        OpenAI API 키
--grid-search         Grid Search 수행 여부
--save-embeddings     임베딩 저장 경로 (.npy)
--load-embeddings     임베딩 로드 경로 (.npy)
```

## 출력 파일

- `topic_info.xlsx`: 토픽 정보 (토픽별 키워드, 문서 수 등)
- `documents_with_topics.xlsx`: 문서별 토픽 할당 결과
- `bertopic_model/`: 학습된 BERTopic 모델
- `topic_hierarchy.html`: 토픽 계층 구조 시각화 (옵션)
- `grid_search_results.xlsx`: Grid Search 결과 (--grid-search 사용 시)

## 설정 커스터마이징

`src/config.py` 파일을 수정하여 설정을 변경할 수 있습니다:

```python
# src/config.py

class Config:
    # 임베딩 모델 변경
    EMBEDDING_MODEL = "text-embedding-3-large"

    # LLM 모델 변경
    LLM_MODEL = "gpt-4o"

    # 최소 토픽 크기 조정
    MIN_TOPIC_SIZE = 15

    # Grid Search 파라미터 조정
    UMAP_GRID = [
        {'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0},
        {'n_neighbors': 30, 'n_components': 10, 'min_dist': 0.1},
    ]

    # 커스텀 불용어 추가
    CUSTOM_STOPWORDS = ['word1', 'word2', 'word3']
```

## 주의사항

1. **OpenAI API 키**: 사용하려면 유효한 OpenAI API 키가 필요합니다.
2. **임베딩 비용**: 문서가 많을 경우 OpenAI API 비용이 발생할 수 있습니다.
3. **메모리**: 대용량 데이터셋의 경우 충분한 메모리가 필요합니다.
4. **Grid Search**: 많은 시간이 소요될 수 있습니다. 소규모 샘플로 먼저 테스트하세요.

## 라이선스

이 프로젝트는 자유롭게 사용할 수 있습니다.

## 문의

문제가 발생하거나 개선 제안이 있으시면 이슈를 등록해주세요.
