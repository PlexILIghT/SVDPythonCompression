from PIL import Image
import numpy as np

def compress_image(image_path, rank_k):
    """_summary_

    Args:
        image_path (_type_): _description_
        rank_k (_type_): _description_

    Returns:
        _type_: _description_
    """
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float64)
    
    channels = [img_array[:, :, i] for i in range(3)]
    
    compressed_channels = []
    original_size = 0
    compressed_size = 0

    for channel in channels:
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        
        U_k = U[:, :rank_k]
        S_k = S[:rank_k]
        Vt_k = Vt[:rank_k, :]
        
        compressed_channel = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
        compressed_channels.append(compressed_channel)
        
        original_size += U.size + S.size + Vt.size
        compressed_size += U_k.size + S_k.size + Vt_k.size
    
    compressed_img_array = np.stack(compressed_channels, axis=2)
    compressed_img_array = np.clip(compressed_img_array, 0, 255).astype('uint8')
    compressed_img = Image.fromarray(compressed_img_array)
    
    
    compression_ratio = compressed_size / original_size
    scale_factor = 1
    reduce_resolution = input("Уменьшить разрешение изображения? ([Y]ay/[N]ay): ").strip().lower()
    
    if reduce_resolution == "yay" or reduce_resolution == "y":
        auto_scale = input("Вычислить масштаб автоматически? ([Y]ay/[N]ay): ").strip().lower()
        
        if auto_scale == "yay" or auto_scale == "y":
            scale_factor = compression_ratio
            print(f"Автоматически выбранный scale_factor: {scale_factor:.2f}")
        else:
            scale_factor = float(input("Введите коэффициент уменьшения разрешения (например, 0.5): "))
    print("asdasd", compressed_img.size[0]*scale_factor, compressed_img.size[1]*scale_factor)
    reduced_width = int(compressed_img.size[0]*scale_factor*4)
    reduced_height = int(compressed_img.size[1]*scale_factor*4)
    compressed_img = compressed_img.resize((reduced_height, reduced_width), Image.Resampling.LANCZOS)

    compressed_img.save("compressed_image.jpg")
    
    return compression_ratio



image_path = "45612000_WGHB8RSyaa5grd7.jpg"

if __name__=="__main__":
    
    rank_k = 0
    print("Ввод ранга (Обычно читабельная картинка получается при k >= 150):\n")
    rank_k = int(input())
        
    try:
        compression = compress_image(image_path, rank_k)
        print(f"Процент сжатия: {100 * compression:.2f}%")
    except Exception as e:
        print(e)