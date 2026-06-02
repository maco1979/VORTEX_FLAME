"BOZAK AR-4 + AlphaTheta Euphonia + MODEL9500BW — 三重声学电路数据增强层 (向量化版)"
import torch
import torch.nn as nn
import math
import random

# ============================================================
# 精确电路参数库
# ============================================================

BOZAK_AR4_PARAMS = {
    "channel_isolator": {"freq_low": 400, "freq_high": 4000, "gain_range_db": (-60, 8)},
    "master_isolator": {"freq_low": 300, "freq_high": 5000, "gain_range_db": (-60, 8)},
    "carbon_resistor": {"alpha": 0.001},
    "transistor_saturation": {"thd_at_1khz": 0.002, "even_harmonic_ratio": 0.70},
    "transformer_saturation": {"drive": 4.0, "knee": 0.7, "compression": 0.05},
    "polyester_capacitor": {"tan_delta": 0.001, "cutoff_freq": 28000},
    "analog_noise": {"snr_db": 95},
}

EUPHONIA_PARAMS = {
    "rupert_neve_tx": {
        "h2_ratio": 0.015, "h3_ratio": 0.003, "h5_ratio": 0.0005,
        "transformer_drive": 0.08, "magnetic_soft_knee": 0.6,
        "phase_shift_deg_at_20khz": 3.0, "output_impedance": 50,
    },
    "master_isolator": {"freq_low": 300, "freq_high": 4000, "gain_range_db": (-36, 8)},
    "riaa_phono": {"t1_us": 75, "t2_us": 318, "t3_us": 3180},
    "dsp_pipeline": {"sample_rate": 96000, "bit_depth": 32, "float_precision": 64, "internal_headroom_db": 24},
    "noise_floor": {"snr_line_db": 105, "snr_phono_db": 88, "snr_digital_db": 107},
    "fx_send_return": {"send_impedance": 600, "return_impedance": 20000, "send_level_pad_db": -6},
    "dynamic_range_db": 110,
}

MODEL9500BW_PARAMS = {
    "five_band_eq": {"frequencies": [75, 300, 1000, 3000, 10000], "gain_range_db": (-12, 12), "q_values": [0.7, 0.9, 1.0, 1.0, 0.8]},
    "master_isolator_3band": {"freq_low": 300, "freq_high": 3000, "gain_range_db": (-18, 18)},
    "jfet_preamp": {"thd_at_1khz": 0.0003},
    "output_stage": {"rca_impedance": 600, "xlr_impedance": 50},
}

# ============================================================
# 预计算 Mel 频域工具 (一次性向量化)
# ============================================================

def _mel_bin_for_freq(freq_hz, sample_rate=22050, n_mels=128):
    if freq_hz <= 0:
        return 0
    mel = 2595.0 * math.log10(1.0 + freq_hz / 700.0)
    max_mel = 2595.0 * math.log10(1.0 + sample_rate / 2 / 700.0)
    return int(round(mel / max_mel * (n_mels - 1)))

def _precompute_isolator_gains(freq_low, freq_high, n_mels=128, sample_rate=22050):
    bin_low = _mel_bin_for_freq(freq_low, sample_rate, n_mels)
    bin_high = _mel_bin_for_freq(freq_high, sample_rate, n_mels)
    template = torch.ones(n_mels)
    for i in range(n_mels):
        if i <= bin_low:
            blend = max(0.0, min(1.0, (bin_low - i) / max(1, bin_low * 0.3)))
            template[i] = blend
        elif i >= bin_high:
            blend = max(0.0, min(1.0, (i - bin_high) / max(1, (n_mels - bin_high) * 0.3)))
            template[i] = blend
        else:
            template[i] = 0.0
    return template

def _precompute_capacitor_rolloff(cutoff_hz, n_mels=128, sample_rate=22050):
    cutoff_bin = _mel_bin_for_freq(cutoff_hz, sample_rate, n_mels)
    rolloff = torch.ones(n_mels)
    for i in range(cutoff_bin, n_mels):
        t = (i - cutoff_bin) / max(1, n_mels - cutoff_bin)
        rolloff[i] = 1.0 - 0.15 * t * t
    return rolloff

def _precompute_eq_shapes(frequencies, q_values, n_mels=128, sample_rate=22050):
    shapes = []
    for freq, Q in zip(frequencies, q_values):
        shape = torch.zeros(n_mels)
        center = _mel_bin_for_freq(freq, sample_rate, n_mels)
        bw_bins = max(1, int(n_mels / (Q * 2)))
        for i in range(n_mels):
            dist = (i - center) / bw_bins
            shape[i] = math.exp(-0.5 * dist * dist)
        shapes.append(shape)
    return torch.stack(shapes)

def _precompute_harmonic_shape(order, base_ratio, n_mels=128):
    curve = torch.zeros(n_mels)
    for i in range(n_mels):
        norm = i / max(1, n_mels - 1)
        curve[i] = base_ratio * (1.0 - norm ** (order * 0.5)) * (0.5 + 0.5 * (1 - norm) ** 0.3)
    return curve

def _precompute_riaa_curve(n_mels=128, sample_rate=22050):
    curve = torch.ones(n_mels)
    bin_50 = _mel_bin_for_freq(50, sample_rate, n_mels)
    bin_500 = _mel_bin_for_freq(500, sample_rate, n_mels)
    bin_2122 = _mel_bin_for_freq(2122, sample_rate, n_mels)
    for i in range(n_mels):
        if i <= bin_50:
            curve[i] = 1.0 / 10.0
        elif i <= bin_500:
            curve[i] = 1.0 / (10.0 ** (1 - (i - bin_50) / max(1, bin_500 - bin_50)))
        elif i <= bin_2122:
            curve[i] = 1.0
        else:
            t = (i - bin_2122) / max(1, n_mels - bin_2122)
            curve[i] = 1.0 / (1.0 + t * 0.5)
    return curve

# ============================================================
# 统一增强层 (向量化batch运算)
# ============================================================

class MultiCircuitAugment(nn.Module):
    def __init__(self, n_mels=128, sample_rate=22050, p_augment=0.5):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.p_augment = p_augment

        self.register_buffer("_iso_bozak_chan", _precompute_isolator_gains(400, 4000, n_mels, sample_rate))
        self.register_buffer("_iso_bozak_mast", _precompute_isolator_gains(300, 5000, n_mels, sample_rate))
        self.register_buffer("_iso_euphonia", _precompute_isolator_gains(300, 4000, n_mels, sample_rate))
        self.register_buffer("_iso_9500", _precompute_isolator_gains(300, 3000, n_mels, sample_rate))

        self.register_buffer("_cap_rolloff", _precompute_capacitor_rolloff(28000, n_mels, sample_rate))
        self.register_buffer("_h2_curve", _precompute_harmonic_shape(2, 0.015, n_mels))
        self.register_buffer("_h3_curve", _precompute_harmonic_shape(3, 0.003, n_mels))
        self.register_buffer("_riaa_curve", _precompute_riaa_curve(n_mels, sample_rate))
        self.register_buffer("_eq_shapes_9500", _precompute_eq_shapes(
            [75, 300, 1000, 3000, 10000], [0.7, 0.9, 1.0, 1.0, 0.8], n_mels, sample_rate,
        ))

        self._circuits = ["bozak", "euphonia", "model9500bw"]

    def _apply_isolator(self, x, iso_template):
        gl = random.uniform(-6, 4)
        gm = random.uniform(-6, 4)
        gh = random.uniform(-6, 4)
        blend = iso_template.to(x.device)
        gain_low = 10 ** (gl / 20)
        gain_mid = 10 ** (gm / 20)
        gain_high = 10 ** (gh / 20)
        g = blend * gain_low + (1 - blend - blend) * gain_mid + blend * gain_high
        g = torch.lerp(gain_mid * torch.ones_like(blend),
                       blend * gain_low + (1 - blend) * gain_high,
                       blend / (blend + (1 - blend) + 1e-8))
        g = (1 - blend) * gain_mid + blend * (blend < 0.5).float() * gain_low + blend * (blend >= 0.5).float() * gain_high
        return x * g.reshape(1, self.n_mels, 1)

    def _apply_isolator_v2(self, x, freq_low, freq_high):
        bins_all = torch.arange(self.n_mels, dtype=torch.float32, device=x.device)
        bin_low = float(_mel_bin_for_freq(freq_low, self.sample_rate, self.n_mels))
        bin_high = float(_mel_bin_for_freq(freq_high, self.sample_rate, self.n_mels))
        gl = random.uniform(-6, 4)
        gm = random.uniform(-6, 4)
        gh = random.uniform(-6, 4)
        gain_low_db = 10 ** (gl / 20)
        gain_mid_db = 10 ** (gm / 20)
        gain_high_db = 10 ** (gh / 20)
        dist_low = (bin_low - bins_all) / max(1.0, bin_low * 0.3)
        dist_high = (bins_all - bin_high) / max(1.0, (self.n_mels - bin_high) * 0.3)
        blend_low = torch.clamp(dist_low, 0.0, 1.0)
        blend_high = torch.clamp(dist_high, 0.0, 1.0)
        is_mid = ((bins_all > bin_low) & (bins_all < bin_high)).float()
        g = (is_mid * gain_mid_db +
             (1 - is_mid) * (bins_all <= bin_low).float() * (blend_low * 1.0 + (1 - blend_low) * gain_low_db) +
             (1 - is_mid) * (bins_all >= bin_high).float() * (blend_high * 1.0 + (1 - blend_high) * gain_high_db))
        g = torch.where(is_mid.bool(), gain_mid_db * torch.ones_like(g), g)
        return x * g.reshape(1, self.n_mels, 1)

    def _bozak(self, x):
        if random.random() < 0.5:
            iso_type = random.choice(["channel", "master"])
            fl = 400 if iso_type == "channel" else 300
            fh = 4000 if iso_type == "channel" else 5000
            x = self._apply_isolator_v2(x, fl, fh)
        if random.random() < 0.6:
            alpha = BOZAK_AR4_PARAMS["carbon_resistor"]["alpha"] + random.uniform(-0.0002, 0.0002)
            alpha = max(1e-6, alpha)
            x_centered = x - x.mean(dim=-1, keepdim=True)
            x = x - alpha * x_centered.abs() * x_centered.sign()
        if random.random() < 0.6:
            thd = BOZAK_AR4_PARAMS["transistor_saturation"]["thd_at_1khz"] + random.uniform(-0.0005, 0.0005)
            thd = max(1e-6, thd)
            scale = 1.0 / (1.0 + thd * 10)
            threshold = 0.5 + random.uniform(-0.1, 0.1)
            sat = torch.tanh(x / max(threshold, 0.01)) * threshold * scale
            x = sat + x * (1.0 - scale)
        if random.random() < 0.4:
            knee = BOZAK_AR4_PARAMS["transformer_saturation"]["knee"] + random.uniform(-0.05, 0.05)
            compression = BOZAK_AR4_PARAMS["transformer_saturation"]["compression"] + random.uniform(-0.01, 0.01)
            compression = max(0.0, compression)
            over = torch.clamp(x.abs() - max(knee, 0.01), min=0)
            factor = 1.0 - compression * (over / (over + 0.1))
            x = x * factor
        if random.random() < 0.5:
            snr = BOZAK_AR4_PARAMS["analog_noise"]["snr_db"] + random.uniform(-2, 2)
            power = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-10)
            noise_power = power / (10 ** (max(snr, 10) / 10))
            x = x + torch.randn_like(x) * noise_power.sqrt()
        if random.random() < 0.5:
            magnitude = random.uniform(0.5, 1.5)
            rolloff = 1.0 - (1.0 - self._cap_rolloff.to(x.device)) * magnitude
            x = x * rolloff.reshape(1, self.n_mels, 1)
        return x

    def _euphonia(self, x):
        if random.random() < 0.5:
            h2 = self._h2_curve.to(x.device).reshape(1, self.n_mels, 1) * random.uniform(0.7, 1.3)
            h3 = self._h3_curve.to(x.device).reshape(1, self.n_mels, 1) * random.uniform(0.7, 1.3)
            h2_component = x.abs() * h2
            h3_component = x.abs().pow(1.5) * h3 * x.sign()
            drive = EUPHONIA_PARAMS["rupert_neve_tx"]["transformer_drive"] * random.uniform(0.8, 1.2)
            x = x + drive * (h2_component + h3_component)
        if random.random() < 0.4:
            knee = EUPHONIA_PARAMS["rupert_neve_tx"]["magnetic_soft_knee"] + random.uniform(-0.05, 0.05)
            knee = max(0.01, knee)
            x = x / (1.0 + (x.abs() / knee).pow(2)).sqrt()
        if random.random() < 0.5:
            x = self._apply_isolator_v2(x, 300, 4000)
        if random.random() < 0.3:
            curve = self._riaa_curve.to(x.device).reshape(1, self.n_mels, 1)
            magnitude = random.uniform(0.5, 1.5)
            x = x * (1.0 + (curve - 1.0) * magnitude)
        if random.random() < 0.3:
            ceiling = 1.0 + random.uniform(0, 0.15)
            x = torch.tanh(x / ceiling) * ceiling
        if random.random() < 0.4:
            noise_db = EUPHONIA_PARAMS["noise_floor"]["snr_line_db"] + random.uniform(-3, 3)
            pwr = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-10)
            npwr = pwr / (10 ** (max(noise_db, 10) / 10))
            x = x + torch.randn_like(x) * npwr.sqrt()
        return x

    def _model9500bw(self, x):
        if random.random() < 0.5:
            shapes = self._eq_shapes_9500.to(x.device)
            gains = (torch.rand(5, device=x.device) * 24 - 12) * (torch.rand(5, device=x.device) * 0.5 + 0.5)
            gains_db = 10 ** (gains / 20)
            eq_effect = (shapes * gains_db.reshape(-1, 1)).sum(dim=0).reshape(1, self.n_mels, 1)
            x = x * (0.7 + 0.3 * eq_effect)
        if random.random() < 0.5:
            x = self._apply_isolator_v2(x, 300, 3000)
        if random.random() < 0.6:
            thd = MODEL9500BW_PARAMS["jfet_preamp"]["thd_at_1khz"] + random.uniform(-0.0001, 0.0001)
            thd = max(1e-6, thd)
            x_centered = x - x.mean(dim=-1, keepdim=True)
            h2_strength = thd * 0.7 * random.uniform(0.7, 1.3)
            h3_strength = thd * 0.2 * random.uniform(0.7, 1.3)
            h2 = x_centered.abs() * h2_strength
            h3 = x_centered.abs().pow(1.5) * x_centered.sign() * h3_strength
            x = x + h2 + h3
        if random.random() < 0.3:
            noise_floor_db = 120 + random.uniform(-5, 5)
            pwr = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-10)
            npwr = pwr / (10 ** (max(noise_floor_db, 10) / 10))
            x = x + torch.randn_like(x) * npwr.sqrt()
        return x

    def forward(self, mel, force_augment=False):
        if not force_augment and random.random() > self.p_augment:
            return mel, {"augmented": False}

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        circuit = random.choice(self._circuits)
        if circuit == "bozak":
            mel = self._bozak(mel)
        elif circuit == "euphonia":
            mel = self._euphonia(mel)
        else:
            mel = self._model9500bw(mel)

        if squeeze:
            mel = mel.squeeze(0)
        return mel, {"augmented": True, "circuit": circuit}

    def get_probe_params(self):
        return {
            "bozak": {"channel_iso_low": 400, "channel_iso_high": 4000},
            "euphonia": {"rupert_neve_h2": 0.015, "snr_line": 105},
            "model9500bw": {"five_band_freqs": [75, 300, 1000, 3000, 10000]},
        }


def build_audio_circuit_augment(n_mels=128, sample_rate=22050, p_augment=0.5):
    return MultiCircuitAugment(n_mels, sample_rate, p_augment)
