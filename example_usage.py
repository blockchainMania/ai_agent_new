"""
간단한 사용 예시

이 스크립트는 BERTopic 모듈을 Python 코드에서 직접 사용하는 방법을 보여줍니다.
"""

import os
import pandas as pd
from src.preprocessing import TextPreprocessor
from src.topic_model import BERTopicModel
from src.metrics import TopicMetrics
from src.config import Config


def main():
    # ===========================
    # 1. 설정
    # ===========================
    print("=" * 60)
    print("BERTopic 토픽 모델링 예시")
    print("=" * 60)

    # API 키 설정 (환경변수에서 가져오기)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("export OPENAI_API_KEY='your-api-key' 명령으로 설정하세요.")
        return

    Config.OPENAI_API_KEY = api_key

    # ===========================
    # 2. 데이터 로드
    # ===========================
    print("\n[1/5] 데이터 로드")

    # 예시: Excel 파일에서 데이터 읽기
    # df = pd.read_excel("data.xlsx")
    # documents = df['번역'].tolist()

    # 테스트용 샘플 데이터
    documents = [
        "This invention relates to a refrigerator with improved cooling system.",
        "The patent describes a new method for hydrogen storage in fuel cells.",
        "An apparatus for controlling temperature in a refrigeration chamber.",
        "Novel fuel cell stack design with enhanced durability.",
        "Refrigerator door assembly with automatic closing mechanism.",
    ] * 10  # 50개 문서 생성

    print(f"  - 총 {len(documents)}개 문서")

    # ===========================
    # 3. 전처리
    # ===========================
    print("\n[2/5] 텍스트 전처리")

    preprocessor = TextPreprocessor(
        custom_stopwords=Config.CUSTOM_STOPWORDS
    )
    processed_docs = preprocessor.preprocess(documents)

    print(f"  - 전처리 완료")
    print(f"  - 예시: {processed_docs[0][:100]}...")

    # ===========================
    # 4. 임베딩 생성
    # ===========================
    print("\n[3/5] 임베딩 생성")

    model = BERTopicModel(
        openai_api_key=Config.OPENAI_API_KEY,
        embedding_model=Config.EMBEDDING_MODEL,
        llm_model=Config.LLM_MODEL,
        min_topic_size=Config.MIN_TOPIC_SIZE
    )

    embeddings = model.create_embeddings(
        documents=processed_docs,
        batch_size=Config.EMBEDDING_BATCH_SIZE,
        show_progress=True
    )

    print(f"  - 임베딩 shape: {embeddings.shape}")

    # ===========================
    # 5. 토픽 모델링 (Grid Search 없이)
    # ===========================
    print("\n[4/5] BERTopic 모델 학습")

    # 기본 파라미터 사용
    umap_params = Config.UMAP_GRID[0]
    hdbscan_params = {
        'min_cluster_size': 5,  # 샘플 데이터가 작으므로 줄임
        'min_samples': 3,
        'cluster_selection_epsilon': 0.05,
        'metric': 'euclidean',
        'prediction_data': False
    }

    topic_model = model.fit(
        documents=processed_docs,
        embeddings=embeddings,
        umap_params=umap_params,
        hdbscan_params=hdbscan_params,
        verbose=True
    )

    print("  - 모델 학습 완료")

    # ===========================
    # 6. 결과 분석
    # ===========================
    print("\n[5/5] 결과 분석")

    # 토픽 정보
    topic_info = topic_model.get_topic_info()
    print(f"\n토픽 정보:")
    print(topic_info[['Topic', 'Count', 'Name']])

    # 평가 지표
    metrics = TopicMetrics.calculate_metrics(topic_info)
    TopicMetrics.print_metrics(metrics)

    # 문서별 토픽 할당
    print(f"\n문서별 토픽 (처음 5개):")
    for i, (doc, topic) in enumerate(zip(documents[:5], topic_model.topics_[:5])):
        print(f"  문서 {i+1}: 토픽 {topic}")
        print(f"    내용: {doc[:60]}...")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
