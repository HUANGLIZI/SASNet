import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt

def look(src):
    plt.imshow(src)
    plt.show()


class ActivationsAndGradients:
    # �Զ�����__call__()��������ȡ���򴫲���������A�ͷ��򴫲����ݶ�A'
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers: # ��������ߴ�����ľ���㣬���Ǵ���һ��Ҳ����
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation
                )
            )
        if hasattr(target_layer, 'register_full_backward_hook'):
        #hasattr(object,name)����ֵ:��������и����Է���True,���򷵻�False
            self.handles.append(
                target_layer.register_full_backward_hook(self.save_gradient))
        else:
            self.handles.append(
                target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, model, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients #���򴫲����ݶ�A��������ǰ�������ĵ������������෴

    def __call__(self, x):
        #�Զ����õ�__call__����
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
            # handleҪ��ʱ�Ƴ�������Ȼ��ռ�ù����ڴ�


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            pass
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)
        # ȡ�����򴫲���������A�ͷ��򴫲����ݶ�A'
    @staticmethod
    # @staticmethod��@classmethodֻ�ǽ�һ����������һ������£�����ʱ�����ٴ���һ��������ã�����һ�������ڵ���Ҳ���ԣ�
    # ����class A����һ��@staticmethod��������f()������ʱ�Ϳ���ֱ��A.f()���ɣ���ȻҲ����A().f()��
    # @classmethod��@staticmethod�������ڵ�һ���������Ĳ�����cls���ࣩ��
    def get_loss(output, target):
        loss = 0
        # ���·����������ѡ��һ��
        loss = output.mul(target) # 1)�鿴ģ��Ԥ����ȷ�Ĳ���-->�����Ķ�Ӧ�����
        _loss = loss.detach().cpu()
        _loss = _loss.squeeze(0).squeeze(0)
        # look(_loss)

        # �����������ڻ������-->.mm()
        # loss = output # 2)�鿴Ԥ��Ĳ���;
        # loss = output.mul(torch.where(target==1, 0, 1)) # 3)�鿴Ԥ�����Ĳ���;
        # loss = target #���ܻش�gt_tensor����Ϊ��ʱ����û��grad��Ҳû��grad_fn������Ϊrequires_grad��False
        return loss

    @staticmethod
    def get_cam_weights(grads): #GAPȫ��ƽ���ػ�
        return np.mean(grads, axis=(2,3), keepdims=True)

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2) #����д������Ȥ�����Լ�¼һ��
        return width, height

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads) #���ݶ�ͼ����ȫ��ƽ���ػ�
        weighted_activations = weights * activations #��ԭ�������Ȩ��
        cam = weighted_activations.sum(axis=1)
        return cam

    @staticmethod
    def scale_cam_img(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img) #��ȥ��Сֵ
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv.resize(img, target_size) #ע�⣺cv2.resize(src, (width, height))��width��heightǰ����ʽӦע��
            result.append(img)
        result = np.float32(result)
        return result

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations] #������ȫ����Ϊndarray��ʽ
        grads_list = [a.cpu().data.numpy() for a in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # һ��һ������ͼ���ݶȶ�Ӧ�Ŵ���
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam<0] = 0 #ReLU
            scaled = self.scale_cam_img(cam, target_size) #��CAMͼ���ŵ�ԭͼ��С��Ȼ����ԭͼ���ӣ��⿼�ǵ�����ͼ����С�ڻ����ԭͼ���
            cam_per_target_layer.append(scaled[:, None, :]) # ��None��ע��λ�ü���һ��ά�ȣ��൱��scaled.unsqueeze(1)
        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_layer):
        cam_per_layer = np.concatenate(cam_per_layer, axis=1) #��channelsά�Ƚ��жѵ�����û������ӵĴ���
        cam_per_layer = np.maximum(cam_per_layer, 0) #np.maximum��(a, b) a��b������λ�Ƚ�ȡ�����
        result = np.mean(cam_per_layer, axis=1) #��channelsά����ƽ����ѹ�����ά�ȣ���ά�ȷ���Ϊ1
        return self.scale_cam_img(result)

    def __call__(self, input_tensor, target): #����������Զ�����__call__()����
        # �����target����Ŀ���gt��˫��Ե��
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        # ���򴫲���������������ActivationsAndGradients������__call__()������ִ��self.model(x)
        # ע�������outputδ��softmax�����Ե����������ʱ��һ��Ҫ������ṹ�е����һ�㼤�����ע�͵�
        # һ��Ҫע�͵�������
        output = self.activations_and_grads(input_tensor)[0]
        _output = output.detach().cpu()
        _output=_output.squeeze(0).squeeze(0)

        self.model.zero_grad()
        # ���loss�ش���Grad-CAM���µĺ���˼�룬Դ�����Ƿ����������Խ�δ��softmax�ĸ���Ԥ������Ϊloss���򴫲���
        # Ȼ�󽫻ش��õ����ݶ�A'�����ݶ�����˵���ڸ÷����иò�������ṹ�𵽵��������Ԥ��Ĳ���չʾ����������������Ԥ��ʱ��ע����
        loss = self.get_loss(output, target)
        print('loss.shape',loss.shape)
        print('target.shape',target.shape)
        loss.backward(torch.ones_like(target), retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor) #�õ�ÿһ���CAM�����ص���һ���б�
        return self.aggregate_multi_layers(cam_per_layer) #��һ�����б����channelsά����ƽ��ѹ������ά�ȴ����Ϊ1

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv.COLORMAP_JET) -> np.ndarray:
    heatmap = cv.applyColorMap(np.uint8(255 * mask), colormap) #��cam�Ľ��ת��α��ɫͼƬ
    if use_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB) #ʹ��opencv�����󣬵õ���һ�㶼��BGR��ʽ����Ҫת��ΪRGB��ʽ
        # OpenCV��ͼ���������ݸ�ʽ��numpy��ndarray���ݸ�ʽ����BGR��ʽ��ȡֵ��Χ��[0,255].
    heatmap = np.float32(heatmap) / 255. #���ŵ�[0,1]֮��

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255*cam)



















