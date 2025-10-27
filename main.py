"""
BERTopic 토픽 모델링 메인 스크립트

사용 예시:
    python main.py --input data.xlsx --output results/
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing import TextPreprocessor
from src.topic_model import BERTopicModel
from src.metrics import TopicMetrics
from src.config import Config


def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(
        description='BERTopic을 이용한 토픽 모델링'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='입력 엑셀 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='출력 디렉토리 경로'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='번역',
        help='분석할 텍스트 컬럼명'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API 키 (환경변수 OPENAI_API_KEY로도 설정 가능)'
    )
    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Grid Search 수행 여부'
    )
    parser.add_argument(
        '--save-embeddings',
        type=str,
        default=None,
        help='임베딩 저장 경로 (.npy)'
    )
    parser.add_argument(
        '--load-embeddings',
        type=str,
        default=None,
        help='임베딩 로드 경로 (.npy)'
    )

    args = parser.parse_args()

    # API 키 설정
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAI API 키를 제공해주세요. "
            "--api-key 옵션 또는 OPENAI_API_KEY 환경변수로 설정할 수 있습니다."
        )

    Config.OPENAI_API_KEY = api_key

    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BERTopic 토픽 모델링 시작")
    print("=" * 60)

    # 1. 데이터 로드
    print(f"\n[1/6] 데이터 로드: {args.input}")
    df = pd.read_excel(args.input)
    print(f"  - 총 {len(df)}개 문서")

    if args.text_column not in df.columns:
        raise ValueError(
            f"컬럼 '{args.text_column}'을 찾을 수 없습니다. "
            f"사용 가능한 컬럼: {list(df.columns)}"
        )

    documents = df[args.text_column].tolist()

    # 2. 전처리
    print(f"\n[2/6] 텍스트 전처리")
    preprocessor = TextPreprocessor(
        custom_stopwords=Config.CUSTOM_STOPWORDS
    )
    processed_docs = preprocessor.preprocess(documents)
    print(f"  - 전처리 완료")

    # 3. 임베딩 생성 또는 로드
    if args.load_embeddings:
        print(f"\n[3/6] 임베딩 로드: {args.load_embeddings}")
        embeddings = np.load(args.load_embeddings)
        print(f"  - 임베딩 shape: {embeddings.shape}")
    else:
        print(f"\n[3/6] 임베딩 생성")
        model = BERTopicModel(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.EMBEDDING_MODEL,
            llm_model=Config.LLM_MODEL,
            min_topic_size=Config.MIN_TOPIC_SIZE
        )
        embeddings = model.create_embeddings(
            processed_docs,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            show_progress=Config.SHOW_PROGRESS
        )
        print(f"  - 임베딩 shape: {embeddings.shape}")

        # 임베딩 저장
        if args.save_embeddings:
            np.save(args.save_embeddings, embeddings)
            print(f"  - 임베딩 저장: {args.save_embeddings}")

    # 4. Grid Search (옵션)
    if args.grid_search:
        print(f"\n[4/6] Grid Search 수행")
        model = BERTopicModel(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.EMBEDDING_MODEL,
            llm_model=Config.LLM_MODEL,
            min_topic_size=Config.MIN_TOPIC_SIZE
        )

        best_params, grid_results = model.grid_search(
            documents=processed_docs,
            embeddings=embeddings,
            sample_size=Config.SAMPLE_SIZE,
            umap_grid=Config.UMAP_GRID,
            hdbscan_grid=Config.HDBSCAN_GRID,
            max_noise=Config.MAX_NOISE_RATIO,
            max_topics=Config.MAX_TOPICS,
            seed=Config.RANDOM_SEED,
            show_progress=Config.SHOW_PROGRESS
        )

        # Grid Search 결과 저장
        grid_results.to_excel(
            output_dir / "grid_search_results.xlsx",
            index=False
        )
        print(f"  - Grid Search 결과 저장: {output_dir / 'grid_search_results.xlsx'}")

        # 최적 파라미터
        umap_params = best_params['umap_params']
        hdbscan_params = {
            'min_cluster_size': best_params['min_cluster_size'],
            'min_samples': best_params['min_samples'],
            'cluster_selection_epsilon': best_params['epsilon'],
            'metric': 'euclidean',
            'prediction_data': False
        }
    else:
        print(f"\n[4/6] Grid Search 생략 (기본 파라미터 사용)")
        umap_params = Config.UMAP_GRID[0]
        hdbscan_params = {
            'min_cluster_size': 20,
            'min_samples': 6,
            'cluster_selection_epsilon': 0.05,
            'metric': 'euclidean',
            'prediction_data': False
        }

    # 5. 전체 데이터로 모델 학습
    print(f"\n[5/6] BERTopic 모델 학습")
    model = BERTopicModel(
        openai_api_key=Config.OPENAI_API_KEY,
        embedding_model=Config.EMBEDDING_MODEL,
        llm_model=Config.LLM_MODEL,
        min_topic_size=Config.MIN_TOPIC_SIZE
    )

    topic_model = model.fit(
        documents=processed_docs,
        embeddings=embeddings,
        umap_params=umap_params,
        hdbscan_params=hdbscan_params,
        verbose=Config.VERBOSE
    )

    # 평가 지표 출력
    topic_info = topic_model.get_topic_info()
    metrics = TopicMetrics.calculate_metrics(topic_info)
    TopicMetrics.print_metrics(metrics)

    # 6. 결과 저장
    print(f"\n[6/6] 결과 저장")

    # 토픽 정보 저장
    topic_info.to_excel(
        output_dir / "topic_info.xlsx",
        index=False
    )
    print(f"  - 토픽 정보: {output_dir / 'topic_info.xlsx'}")

    # 문서별 토픽 할당
    df['topic'] = topic_model.topics_
    df.to_excel(
        output_dir / "documents_with_topics.xlsx",
        index=False
    )
    print(f"  - 문서별 토픽: {output_dir / 'documents_with_topics.xlsx'}")

    # 모델 저장
    model_path = str(output_dir / "bertopic_model")
    model.save_model(model_path)
    print(f"  - 모델 저장: {model_path}")

    # 시각화 (HTML로 저장)
    try:
        fig = topic_model.visualize_hierarchy()
        fig.write_html(str(output_dir / "topic_hierarchy.html"))
        print(f"  - 계층 구조: {output_dir / 'topic_hierarchy.html'}")
    except Exception as e:
        print(f"  - 시각화 실패: {e}")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
