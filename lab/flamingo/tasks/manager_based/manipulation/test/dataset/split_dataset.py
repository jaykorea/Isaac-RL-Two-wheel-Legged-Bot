import zarr
import numpy as np
import os
import shutil
from tqdm import tqdm

def split_zarr_file(
    input_path: str,
    output_path1: str,
    output_path2: str,
    split_ratio: float = 0.5,
    read_batch_size: int = 100000  # 한 번에 처리할 데이터 수 (메모리 절약용)
):
    """
    대용량 Zarr 파일을 메모리 폭발 없이 청크 단위로 읽어 두 개의 파일로 분할합니다.
    """
    # 1. 원본 열기 (Read Mode)
    print(f"[INFO] Opening Source: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found: {input_path}")

    src_group = zarr.open_group(input_path, mode='r')
    
    if "transitions" not in src_group:
        raise ValueError("Source zarr does not have 'transitions' dataset.")
    
    src_ds = src_group["transitions"]
    total_len = src_ds.shape[0]
    
    # 2. 분기점 계산
    split_idx = int(total_len * split_ratio)
    
    print(f"[INFO] Total samples: {total_len}")
    print(f"[INFO] Split Index: {split_idx}")
    print(f"       File 1: 0 ~ {split_idx} ({split_idx} samples)")
    print(f"       File 2: {split_idx} ~ {total_len} ({total_len - split_idx} samples)")

    # 3. 작업 정의 (파일명, 시작 인덱스, 끝 인덱스)
    tasks = [
        (output_path1, 0, split_idx),
        (output_path2, split_idx, total_len)
    ]

    # 4. 분할 복사 시작
    for out_path, start_idx, end_idx in tasks:
        # 이미 존재하면 삭제 후 생성 (안전장치)
        if os.path.exists(out_path):
            print(f"[WARN] Removing existing output: {out_path}")
            shutil.rmtree(out_path)

        print(f"\n[PROCESS] Creating {out_path} ...")
        
        # (1) 타겟 그룹 생성
        dst_group = zarr.open_group(out_path, mode='w')
        
        # (2) 메타데이터(Attributes) 복사 (obs_dim, depth_shape 등 필수 정보)
        # 이 부분이 없으면 train.py에서 설정 오류가 납니다.
        dst_group.attrs.update(src_group.attrs)
        
        # (3) 데이터셋 틀 생성 (내용은 아직 없음)
        # Source와 동일한 압축 방식(Compressor)과 청크 크기(Chunks)를 사용
        dst_ds = dst_group.create_dataset(
            "transitions",
            shape=(0,),            # 0부터 시작해서 append
            maxshape=(None,),      # 무한 확장 가능하도록 설정
            dtype=src_ds.dtype,
            chunks=src_ds.chunks,
            compressor=src_ds.compressor
        )
        
        # (4) 스트리밍 복사 (Batch 단위)
        # start_idx부터 end_idx까지 read_batch_size만큼씩 끊어서 읽고 씁니다.
        current_ptr = start_idx
        total_to_copy = end_idx - start_idx
        
        pbar = tqdm(total=total_to_copy, unit="steps", dynamic_ncols=True)
        
        while current_ptr < end_idx:
            # 이번에 읽을 끝 지점 (end_idx를 넘지 않도록)
            read_end = min(current_ptr + read_batch_size, end_idx)
            
            # [핵심] 슬라이싱: 디스크에서 필요한 만큼만 RAM에 로드 (여기서 압축 풀림)
            # 예: 10만 개만 읽으므로 메모리는 수백 MB만 사용
            chunk_data = src_ds[current_ptr : read_end]
            
            # 새 파일에 쓰기 (다시 압축되어 저장됨)
            dst_ds.append(chunk_data)
            
            # 진행상황 업데이트
            count = read_end - current_ptr
            pbar.update(count)
            current_ptr = read_end
            
        pbar.close()
        print(f"[DONE] Saved to {out_path} (Final shape: {dst_ds.shape})")

    print("\n[SUCCESS] All splitting jobs finished.")

if __name__ == "__main__":
    # ==========================================
    # [설정] 여기에 파일 경로를 입력하세요
    # ==========================================
    INPUT_ZARR = "./lift_pick_and_lift_sm_20251221_030635/transitions"      # 원본 파일 경로 (수정 필요)
    OUT_1 = "./lift_pick_and_lift_sm_20251221_030635_1/transitions"           # 저장할 경로 1
    OUT_2 = "./lift_pick_and_lift_sm_20251221_030635_2/transitions"           # 저장할 경로 2
    
    # 실행
    split_zarr_file(INPUT_ZARR, OUT_1, OUT_2)