import torch
import torch.nn as nn
from types import SimpleNamespace


class Channel(nn.Module):
    """
    信道模型类，支持：
    - 无噪声信道 ('none' 或 0)
    - AWGN信道 ('awgn' 或 1)
    - 瑞利衰落信道 ('rayleigh' 或 2)
    """

    def __init__(self, config):
        """
        初始化信道模型
        Args:
            config: 配置对象 (SimpleNamespace 类型，包含以下属性)
                - channel_type: 信道类型 ('none'/'awgn'/'rayleigh' 或 0/1/2)
                - multiple_snrs: 信噪比 (dB)
                - logger: 日志记录器 (可选，默认为 None)
        """
        super(Channel, self).__init__()
        # 从 config 对象中读取参数 (使用 . 访问，非常直观)
        self.chan_type = config.channel_type
        # self.device = config.device
        
        # 可选的日志记录
        logger = getattr(config, 'logger', None)
        if logger:
            logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                config.channel_type, config.multiple_snrs))

    def gaussian_noise_layer(self, input_layer, std):
        """
        添加复高斯白噪声（AWGN）
        :param input_layer: 输入信号（复数张量）
        :param std: 噪声标准差
        :return: 添加噪声后的信号
        """
        device = input_layer.device
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std):
        """
        瑞利衰落信道
        :param input_layer: 输入信号（复数张量）
        :param std: 噪声标准差
        :return: 经瑞利信道后的信号
        """
        device = input_layer.device
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag

        # 生成瑞利衰落系数 h = sqrt(x² + y²)/sqrt(2)
        h = torch.sqrt(
            torch.normal(mean=0.0, std=1, size=input_layer.shape, device=device) ** 2
            + torch.normal(mean=0.0, std=1, size=input_layer.shape, device=device) ** 2
        ) / torch.sqrt(torch.tensor(2.0, device=device))

        # 瑞利信道模型：y = h * x + n
        return input_layer * h + noise

    def complex_normalize(self, x, power):
        """
        复数信号功率归一化
        :param x: 复数信号（实部+虚部分开的张量）
        :param power: 目标功率
        :return: 归一化后的信号和功率
        """
        pwr = torch.mean(x ** 2) * 2
        out = torch.sqrt(torch.tensor(power, device=x.device)) * x / torch.sqrt(pwr)
        return out, pwr

    def forward(self, input, chan_param, use_avg_pwr=False, avg_pwr_value=1.0):
        """
        前向传播
        :param input: 输入信号（实部+虚部分开的张量）
        :param chan_param: 信道参数（SNR，单位dB）
        :param use_avg_pwr: 是否使用平均功率归一化（布尔开关）
        :param avg_pwr_value: 平均功率数值（仅当use_avg_pwr=True时生效）
        :return: 经过信道后的输出信号
        """
        assert input.numel() % 2 == 0, \
            f"输入总元素数必须为偶数（实部虚部分开存储），当前元素数：{input.numel()}"

        # 1. 功率归一化
        if use_avg_pwr:
            power = 1
            channel_tx = torch.sqrt(torch.tensor(power, device=input.device)) * input \
                         / torch.sqrt(torch.tensor(avg_pwr_value * 2, device=input.device))
            pwr = avg_pwr_value * 2
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)

        # 2. 调整输入形状：实部+虚部合并为复数
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j

        # 3. 通过信道（复数域处理）
        channel_output = self.complex_forward(channel_in, chan_param)

        # 4. 恢复形状：复数拆分为实部+虚部
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)

        # 5. 恢复功率归一化前的尺度
        if self.chan_type in (1, 'awgn', 2, 'rayleigh'):
            if use_avg_pwr:
                return channel_output * torch.sqrt(torch.tensor(avg_pwr_value * 2, device=input.device))
            else:
                return channel_output * torch.sqrt(pwr)
        else:
            return channel_output

    def complex_forward(self, channel_in, chan_param):
        """
        复数域的信道处理
        :param channel_in: 输入信号（复数形式）
        :param chan_param: 信道参数（SNR，单位dB）
        :return: 经信道后的输出信号
        """
        if self.chan_type in (0, 'none'):
            return channel_in

        elif self.chan_type in (1, 'awgn'):
            channel_tx = channel_in
            sigma = torch.sqrt(1.0 / (2 * torch.pow(10, chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx, std=sigma)
            return chan_output

        elif self.chan_type in (2, 'rayleigh'):
            channel_tx = channel_in
            sigma = torch.sqrt(1.0 / (2 * torch.pow(10, chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_tx, std=sigma)
            return chan_output

    def noiseless_forward(self, channel_in):
        """
        无噪声前向传播
        :param channel_in: 输入信号（实部+虚部分开的张量）
        :return: 经无噪声信道后的信号
        """
        channel_tx, _ = self.complex_normalize(channel_in, power=1)
        return channel_tx
    
class Channel1(nn.Module):
    """
    信道模型类，支持：
    - 无噪声信道 ('none' 或 0)
    - AWGN信道 ('awgn' 或 1)
    - 瑞利衰落信道 ('rayleigh' 或 2)
    
    核心优化：
    1. 修复complex_normalize的功率计算逻辑，按单样本（B维度独立）计算功率
    2. 增加数值稳定性（eps），避免除零/梯度NaN
    3. 优化代码结构和注释，保持原有接口完全兼容
    """

    def __init__(self, config):
        """
        初始化信道模型
        Args:
            config: 配置对象 (SimpleNamespace 类型，包含以下属性)
                - channel_type: 信道类型 ('none'/'awgn'/'rayleigh' 或 0/1/2)
                - multiple_snrs: 信噪比 (dB)
                - logger: 日志记录器 (可选，默认为 None)
        """
        super(Channel1, self).__init__()
        self.chan_type = config.channel_type
        self.eps = 1e-8  # 数值稳定性常数，避免除零
        
        # 可选的日志记录
        logger = getattr(config, 'logger', None)
        if logger:
            logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                config.channel_type, config.multiple_snrs))

    def gaussian_noise_layer(self, input_layer, std):
        """
        添加复高斯白噪声（AWGN）
        :param input_layer: 输入信号（复数张量）
        :param std: 噪声标准差
        :return: 添加噪声后的信号
        """
        device = input_layer.device
        # 实部和虚部分别生成高斯噪声，保证复高斯分布
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std):
        """
        瑞利衰落信道
        :param input_layer: 输入信号（复数张量）
        :param std: 噪声标准差
        :return: 经瑞利信道后的信号
        """
        device = input_layer.device
        # 复高斯噪声
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag

        # 生成瑞利衰落系数（服从瑞利分布）
        h_real = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device)
        h_imag = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device)
        h = torch.sqrt(h_real ** 2 + h_imag ** 2) / torch.sqrt(torch.tensor(2.0, device=device))

        # 瑞利信道模型：y = h * x + n
        return input_layer * h + noise

    def complex_normalize(self, x, power):
        """
        复数信号功率归一化（修复维度平均逻辑，单样本独立归一化）
        :param x: 复数信号（实部+虚部分开的张量），形状支持4D (B,C,H,W)
        :param power: 目标功率
        :return: 归一化后的信号和功率
        """
        # 仅对单样本的 C/H/W 维度求平均，保留batch维度独立性
        # keepdim=True 保证广播时维度匹配
        pwr_per_sample = torch.mean(x ** 2, dim=tuple(range(1, x.ndim)), keepdim=True) * 2
        # 数值稳定性：避免除零
        pwr_per_sample = pwr_per_sample.clamp_min(self.eps)
        
        # 计算缩放因子，逐样本独立归一化到目标功率
        scale = torch.sqrt(torch.tensor(power, device=x.device, dtype=x.dtype)) / torch.sqrt(pwr_per_sample)
        out = x * scale
        
        # 兼容原有返回格式：返回标量（全局平均）保持接口不变
        # 注：内部已按单样本归一化，返回的pwr为全局平均功率（兼容原接口）
        pwr_global = torch.mean(pwr_per_sample)
        return out, pwr_global

    def forward(self, input, chan_param, use_avg_pwr=False, avg_pwr_value=1.0):
        """
        前向传播（接口完全不变）
        :param input: 输入信号（实部+虚部分开的张量）
        :param chan_param: 信道参数（SNR，单位dB）
        :param use_avg_pwr: 是否使用平均功率归一化（布尔开关）
        :param avg_pwr_value: 平均功率数值（仅当use_avg_pwr=True时生效）
        :return: 经过信道后的输出信号
        """
        # 校验输入长度为偶数（实部+虚部分开）
        assert input.numel() % 2 == 0, \
            f"输入总元素数必须为偶数（实部虚部分开存储），当前元素数：{input.numel()}"

        # 1. 功率归一化
        if use_avg_pwr:
            power = 1.0
            scale = torch.sqrt(torch.tensor(power, device=input.device, dtype=input.dtype)) / \
                    torch.sqrt(torch.tensor(avg_pwr_value * 2 + self.eps, device=input.device, dtype=input.dtype))
            channel_tx = input * scale
            pwr = avg_pwr_value * 2
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1.0)

        # 2. 调整输入形状：实部+虚部合并为复数
        input_shape = channel_tx.shape
        channel_in_flat = channel_tx.reshape(-1)
        L = channel_in_flat.shape[0]
        # 前半部分为实部，后半部分为虚部
        channel_in = channel_in_flat[:L // 2] + 1j * channel_in_flat[L // 2:]

        # 3. 复数域信道处理
        channel_output = self.complex_forward(channel_in, chan_param)

        # 4. 恢复形状：复数拆分为实部+虚部
        channel_output_flat = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output_flat.reshape(input_shape)

        # 5. 恢复功率归一化前的尺度
        if self.chan_type in (1, 'awgn', 2, 'rayleigh'):
            if use_avg_pwr:
                restore_scale = torch.sqrt(torch.tensor(avg_pwr_value * 2, device=input.device, dtype=input.dtype))
                return channel_output * restore_scale
            else:
                restore_scale = torch.sqrt(torch.tensor(pwr, device=input.device, dtype=input.dtype))
                return channel_output * restore_scale
        else:
            return channel_output

    def complex_forward(self, channel_in, chan_param):
        """
        复数域的信道处理（接口不变）
        :param channel_in: 输入信号（复数形式）
        :param chan_param: 信道参数（SNR，单位dB）
        :return: 经信道后的输出信号
        """
        if not isinstance(chan_param, torch.Tensor):
            chan_param = torch.tensor(chan_param, device=channel_in.device, dtype=channel_in.dtype)
        # 无噪声信道
        if self.chan_type in (0, 'none'):
            return channel_in

        # AWGN信道
        elif self.chan_type in (1, 'awgn'):
            # 计算噪声标准差（基于SNR）
            snr_linear = torch.pow(10.0, chan_param / 10.0)
            sigma = torch.sqrt(1.0 / (2 * snr_linear + self.eps))
            return self.gaussian_noise_layer(channel_in, std=sigma)

        # 瑞利衰落信道
        elif self.chan_type in (2, 'rayleigh'):
            snr_linear = torch.pow(10.0, chan_param / 10.0)
            sigma = torch.sqrt(1.0 / (2 * snr_linear + self.eps))
            return self.rayleigh_noise_layer(channel_in, std=sigma)
        
        # 未知信道类型
        else:
            raise ValueError(f"不支持的信道类型：{self.chan_type}，可选类型：0/none, 1/awgn, 2/rayleigh")

    def noiseless_forward(self, channel_in):
        """
        无噪声前向传播（接口不变）
        :param channel_in: 输入信号（实部+虚部分开的张量）
        :return: 经无噪声信道后的信号
        """
        channel_tx, _ = self.complex_normalize(channel_in, power=1.0)
        return channel_tx