#coding=utf-8
#tensorflow高效数据读取训练
import tensorflow as tf
import cv2

'''
数据打包，转换成tfrecords格式，以便后续高效读取
agrv:
    label_file:train.txt文件格式，每一行：图片路径名   类别标签
    data_root:生成文件data.tfrecords所在路径
    resize：存储图片的尺寸
    
    以dictionary方式保存，并序列化二进制写入文件data.tfrecords中
    
'''
def encode_to_tfrecords(lable_file, data_root, file_name, resize=None):
    writer=tf.python_io.TFRecordWriter(data_root+'/'+file_name)
    num_example=0
    with open(lable_file,'r') as f:
        for l in f.readlines():
            l=l.split()
            image=cv2.imread(data_root+"/"+l[0])
            if resize is not None:
                image=cv2.resize(image,resize)
            height,width,nchannel=image.shape
            label=int(l[1])
 
            example=tf.train.Example(features=tf.train.Features(feature={
                'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'nchannel':tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            serialized=example.SerializeToString() #序列化数据 
            writer.write(serialized)
            num_example+=1
    print lable_file,"数据量：",num_example
    writer.close()
'''    
读取tfrecords文件
argv:
    filename: .tfrecords数据文件
return：
    image：图片image[0]=height
              image[1]=width
              image[3]=nchannel
    label:图片标签
'''
def decode_from_tfrecords(filename,num_epoch=None):
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)#因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    reader=tf.TFRecordReader()
    _,serialized=reader.read(filename_queue)
    example=tf.parse_single_example(serialized,features={
        'height':tf.FixedLenFeature([],tf.int64),
        'width':tf.FixedLenFeature([],tf.int64),
        'nchannel':tf.FixedLenFeature([],tf.int64),
        'image':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
    })
    label=tf.cast(example['label'], tf.int32)
    image=tf.decode_raw(example['image'],tf.uint8)
    image=tf.reshape(image,tf.pack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['nchannel'], tf.int32)]))
    return image,label
'''
根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
argv:
    image:样本图片
    label：样本标签
    batch_size:扩充图片的数量
    crop_size:剪裁尺寸
return:batch_size 图片数量 image和label
'''
def get_batch(image, label, batch_size, crop_size):
        #数据扩充变换
    distorted_image=tf.image.central_crop(image,33./37.)
    distorted_image = tf.random_crop(distorted_image, [crop_size, crop_size, 3])#随机裁剪,通道数
# #     distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转
#     distorted_image = tf.image.random_brightness(distorted_image,max_delta=50)#亮度变化  
#     distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化  
    
    #生成batch
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大
    #保证数据打的足够乱
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,
                                                 num_threads=4,capacity=50000,min_after_dequeue=10000)
   
    # 调试显示
    #tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])

#这个是用于测试阶段，使用的get_batch函数
def get_test_batch(image, label, batch_size,crop_size): #数据扩充变换
    distorted_image=tf.image.central_crop(image,33./37.)    
    distorted_image = tf.random_crop(distorted_image, [crop_size, crop_size, 3])#随机裁剪,通道数
# #     distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转
#     distorted_image = tf.image.random_brightness(distorted_image,max_delta=50)#亮度变化  
#     distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化  
    
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,
                                                 num_threads=4,capacity=50000,min_after_dequeue=10000)
    return images, tf.reshape(label_batch, [batch_size])

