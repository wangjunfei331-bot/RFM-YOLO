import time
from pathlib import Path
from ultralytics import YOLO

def benchmark_on_dataset(model_path, images_dir, batch_size=16, imgsz=640, half=True, warmup_iters=30):
    # 1. 加载模型（不要手动调用 .half()，让 predict 参数控制）
    model = YOLO(model_path)

    # 2. 收集所有图片路径
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(Path(images_dir).rglob(f'*{ext}'))
        img_paths.extend(Path(images_dir).rglob(f'*{ext.upper()}'))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(f'No images found in {images_dir}')
    print(f'Found {len(img_paths)} images')

    # 3. 预热（使用前 warmup_iters 张图像）
    warmup_paths = img_paths[:min(warmup_iters, len(img_paths))]
    for i in range(0, len(warmup_paths), batch_size):
        batch = warmup_paths[i:i+batch_size]
        _ = model.predict(batch, imgsz=imgsz, half=half, batch=len(batch), verbose=False)

    # 4. 正式计时，累加预处理/推理/后处理时间
    total_pre = 0.0
    total_inf = 0.0
    total_post = 0.0
    processed = 0

    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i+batch_size]
        results = model.predict(batch, imgsz=imgsz, half=half, batch=len(batch), verbose=False)
        for r in results:
            total_pre += r.speed['preprocess']
            total_inf += r.speed['inference']
            total_post += r.speed['postprocess']
            processed += 1
        if processed % 500 == 0:
            print(f'Processed {processed}/{len(img_paths)} images...')

    # 5. 计算平均值
    avg_pre = total_pre / processed
    avg_inf = total_inf / processed
    avg_post = total_post / processed
    total_avg = avg_pre + avg_inf + avg_post

    print(f'\n======= Dataset Benchmark =======')
    print(f'Total images: {processed}')
    print(f'Preprocess avg: {avg_pre:.3f} ms/img')
    print(f'Inference avg:   {avg_inf:.3f} ms/img')
    print(f'Postprocess avg: {avg_post:.3f} ms/img')
    print(f'End-to-end avg:  {total_avg:.3f} ms/img')
    return avg_pre, avg_inf, avg_post

if __name__ == '__main__':
    # 请修改为您的实际路径
    model_path = 'runs/detect/train93/weights/best.pt'
    images_dir = '/root/autodl-tmp/AI-TOD/images/val'
    
    benchmark_on_dataset(
        model_path=model_path,
        images_dir=images_dir,
        batch_size=1,
        imgsz=640,
        #half=True,          # 启用 FP16 加速
        half=False,
        warmup_iters=30
    )