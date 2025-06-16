# --- Bloco de Importação de Bibliotecas ---

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pywt

# --- Bloco de Configurações e Constantes Globais ---

COLORS = {
    "navy": "#0E1627",       # Fundo da janela principal
    "prune": "#7F6269",      # Cor para elementos secundários
    "mauve": "#BD8E89",      # Cor para botões e destaques
    "pink": "#E5C5C1",       # Cor de fundo do "app" e elementos claros
    "blush": "#F4E1E0",      # Cor para textos e títulos na interface clara
    "text_dark": "#0E1627",  # Cor para textos sobre fundos claros
    "yellow": "#FFD700"      # Cor para medianas e destaques importantes
}

BANDS = {'theta': [4, 8], 'beta': [13, 30]}
FS = 256

# --- Funções de Análise de Sinal ---

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def analyze_signal_with_fft(signal, fs, bands):
    signal_filtered = apply_bandpass_filter(signal, 0.5, 50, fs)
    n = len(signal_filtered)
    window = np.hanning(n)
    signal_windowed = signal_filtered * window
    yf = np.fft.fft(signal_windowed)
    xf = np.fft.fftfreq(n, 1 / fs)[:n//2]
    psd = (2.0/n * np.abs(yf[0:n//2]))**2
    power_in_bands = {}
    for band_name, (low_freq, high_freq) in bands.items():
        band_indices = np.where((xf >= low_freq) & (xf <= high_freq))[0]
        power_in_bands[band_name] = np.trapz(psd[band_indices], xf[band_indices]) if len(band_indices) > 0 else 0
    beta_power = power_in_bands.get('beta', 0)
    tbr = power_in_bands.get('theta', 0) / beta_power if beta_power > 1e-12 else 0
    return tbr

def analyze_signal_with_wavelet(signal, fs, wavelet='cmor1.5-1.0'):
    freqs_of_interest = np.linspace(1, 50, 100)
    scales = pywt.scale2frequency(wavelet, freqs_of_interest) / (1/fs)
    coeffs, freqs = pywt.cwt(signal, scales, wavelet, 1/fs)
    power = np.abs(coeffs)**2
    times = np.linspace(0, len(signal)/fs, power.shape[1])
    return power, freqs, times

def calculate_spectrogram(signal, fs):
    nperseg = fs
    noverlap = fs // 2
    freqs, times, Sxx = spectrogram(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='density')
    return freqs, times, Sxx

# --- Estrutura da Interface Gráfica ---

class EEGAnalyzerApp(ctk.CTk):
    """
    Classe principal que controla a janela e a navegação entre as "telas" do app.
    """
    def __init__(self):
        super().__init__()
        self.title("Analisador EEG")
        self.geometry("800x950")
        self.configure(fg_color=COLORS["navy"])

        self.phone_frame = ctk.CTkFrame(self, width=750, height=900, fg_color=COLORS["pink"], corner_radius=30)
        self.phone_frame.pack(expand=True, pady=20)
        self.phone_frame.pack_propagate(False)

        self.frames = {}
        self.create_screens()
        self.show_frame("HomeScreen")

    def create_screens(self):
        """Cria todas as telas (frames) e as armazena em um dicionário."""
        self.frames["HomeScreen"] = HomeScreen(self.phone_frame, controller=self)
        self.frames["ResultsScreen"] = ResultsScreen(self.phone_frame, controller=self)

    def show_frame(self, screen_name):
        """Esconde a tela atual e mostra a tela solicitada."""
        for frame in self.frames.values():
            frame.pack_forget()
        frame_to_show = self.frames[screen_name]
        frame_to_show.pack(fill="both", expand=True)

    def run_analysis_and_show_results(self):
        """Orquestra a análise e a transição para a tela de resultados."""
        home_screen = self.frames["HomeScreen"]
        results_screen = self.frames["ResultsScreen"]
        
        home_screen.lbl_status.configure(text="Analisando... Isso pode levar um momento.", text_color=COLORS["prune"])
        self.update_idletasks()
        try:
            loaded_data = results_screen.load_mat_data(home_screen.data_folder_path)
            results_screen.loaded_data = loaded_data
            analysis_results = results_screen.analyze_loaded_data(loaded_data)
            results_screen.analysis_results = analysis_results
            results_screen.plot_all_results()
            self.show_frame("ResultsScreen")
            home_screen.lbl_status.configure(text="Análise Concluída!", text_color=COLORS["mauve"])
        except Exception as e:
            home_screen.lbl_status.configure(text=f"Erro: {e}", text_color="red")
            messagebox.showerror("Erro na Análise", f"Ocorreu um erro:\n{e}")
            import traceback
            traceback.print_exc()

class HomeScreen(ctk.CTkFrame):
    """
    A tela inicial do aplicativo. Contém os controles principais.
    """
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.data_folder_path = ""
        
        ctk.CTkLabel(self, text="🧠🦋", font=ctk.CTkFont(size=60)).pack(pady=(80, 10))
        ctk.CTkLabel(self, text="Analisador EEG", font=ctk.CTkFont(size=28, weight="bold"), text_color=COLORS["text_dark"]).pack()
        ctk.CTkLabel(self, text="Análise de TDAH", font=ctk.CTkFont(size=18), text_color=COLORS["prune"]).pack(pady=(0, 60))
        
        self.btn_select_folder = ctk.CTkButton(self, text="Selecionar Pasta de Dados", corner_radius=15, command=self.on_select_folder_click, fg_color=COLORS["mauve"], text_color=COLORS["blush"], hover_color=COLORS["prune"])
        self.btn_select_folder.pack(pady=20, padx=40, ipady=10, fill="x")
        self.lbl_folder_path = ctk.CTkLabel(self, text="", text_color=COLORS["prune"])
        self.lbl_folder_path.pack(pady=5)
        self.btn_run_analysis = ctk.CTkButton(self, text="INICIAR ANÁLISE", corner_radius=15, command=controller.run_analysis_and_show_results, state="disabled", fg_color=COLORS["prune"], text_color=COLORS["blush"], hover_color=COLORS["mauve"], font=ctk.CTkFont(size=16, weight="bold"))
        self.btn_run_analysis.pack(pady=20, padx=40, ipady=10, fill="x")
        self.lbl_status = ctk.CTkLabel(self, text="Selecione uma pasta para começar", text_color=COLORS["mauve"])
        self.lbl_status.pack(pady=20)
        
    def on_select_folder_click(self):
        self.data_folder_path = filedialog.askdirectory()
        if self.data_folder_path:
            self.lbl_folder_path.configure(text=f"Pasta: ...{os.path.basename(self.data_folder_path)}")
            self.btn_run_analysis.configure(state="normal")
            self.lbl_status.configure(text="Pronto para analisar!")

class ResultsScreen(ctk.CTkFrame):
    """
    A tela que exibe todos os resultados da análise em abas, com a nova funcionalidade.
    """
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.loaded_data = {}
        self.analysis_results = {}

        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=10)
        back_arrow = ctk.CTkButton(header_frame, text="← Voltar", command=lambda: controller.show_frame("HomeScreen"), fg_color="transparent", text_color=COLORS["prune"], hover_color=COLORS["pink"], width=50)
        back_arrow.pack(side="left")
        ctk.CTkLabel(header_frame, text="Resultados da Análise", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLORS["text_dark"]).pack(side="left", expand=True)

        self.tabview = ctk.CTkTabview(self, fg_color=COLORS["pink"])
        self.tabview.pack(pady=10, padx=10, fill="both", expand=True)
        self.tab_group = self.tabview.add("Comparação")
        self.tab_individual = self.tabview.add("Análise de Caso Individual")
        self.tab_average = self.tabview.add("Análises Médias")

    def load_mat_data(self, folder_path):
        data = {'F-ADHD': [], 'F-Ctrl': [], 'M-ADHD': [], 'M-Ctrl': []}
        files_map = {'F-ADHD': 'FADHD.mat', 'F-Ctrl': 'FC.mat', 'M-ADHD': 'MADHD.mat', 'M-Ctrl': 'MC.mat'}
        for group, filename in files_map.items():
            path = os.path.join(folder_path, filename)
            if not os.path.exists(path): raise FileNotFoundError(f"Arquivo não encontrado: {filename}")
            mat = sio.loadmat(path)
            key = [k for k in mat if not k.startswith('__')][0]
            subjects_raw = mat[key].flatten()
            for subj_raw in subjects_raw:
                subj_data = subj_raw[0] if isinstance(subj_raw, np.ndarray) and subj_raw.size > 0 else subj_raw
                if subj_data.ndim == 1: subj_data = subj_data.reshape(1, -1)
                if subj_data.shape[0] > subj_data.shape[1]: subj_data = subj_data.T
                data[group].append(subj_data)
        return data

    def analyze_loaded_data(self, data):
        results = {'tbr_by_group': {}}
        for group, subjects in data.items():
            results['tbr_by_group'][group] = [analyze_signal_with_fft(s[0, :], FS, BANDS) for s in subjects]
        return results

    def _apply_plot_style(self, ax, fig, title):
        fig.patch.set_facecolor(COLORS["pink"])
        ax.set_facecolor(COLORS["pink"])
        ax.set_title(title, color=COLORS["text_dark"], fontsize=12, weight="bold")
        ax.tick_params(axis='both', colors=COLORS["prune"])
        ax.xaxis.label.set_color(COLORS["prune"])
        ax.yaxis.label.set_color(COLORS["prune"])
        for spine in ax.spines.values(): spine.set_edgecolor(COLORS["prune"])
        ax.grid(True, linestyle='--', color=COLORS["mauve"], alpha=0.3)
        return ax

    def plot_all_results(self):
        self.plot_group_comparison()
        self.create_individual_analysis_tab()
        self.update_individual_plots()
        self.create_average_analysis_tab()
        self.plot_average_analyses()

    def plot_group_comparison(self):
        """Plota os boxplots comparativos e adiciona uma legenda explicativa."""
        # Limpa os widgets anteriores da aba para evitar sobreposição
        for widget in self.tab_group.winfo_children(): widget.destroy()
        
        # Cria o frame e o canvas para o gráfico
        canvas_frame = ctk.CTkFrame(self.tab_group, fg_color="transparent")
        canvas_frame.pack(fill="both", expand=True, pady=10)
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Prepara os dados e plota o boxplot (código existente)
        tbr_data = self.analysis_results['tbr_by_group']
        data_to_plot = [tbr_data['F-ADHD'], tbr_data['F-Ctrl'], tbr_data['M-ADHD'], tbr_data['M-Ctrl']]
        labels = ['TDAH (F)', 'Controle (F)', 'TDAH (M)', 'Controle (M)']
        
        box = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.5)
        colors = [COLORS["mauve"], COLORS["prune"], COLORS["mauve"], COLORS["prune"]]
        for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color)
        for median in box['medians']: median.set(color=COLORS['yellow'], linewidth=2)
            
        # Aplica o estilo geral ao gráfico
        self._apply_plot_style(ax, fig, "Razão Teta/Beta por Grupo")
        ax.set_ylabel("TBR")
        
        # --- NOVO: LEGENDA EXPLICATIVA ---
        # Define o texto que aparecerá na legenda
        legend_text = (
            'LEGENDA DO GRÁFICO:\n'
            '----------------------------------\n'
            '--  Linha Amarela: Mediana (valor central)\n'
            '■  Caixa: 50% centrais dos dados\n'
            '─  Linhas Pretas: Valor Mínimo e Máximo\n'
            '○  Círculos: Outliers (valores atípicos)'
        )
        
        # Define as propriedades da caixa de texto (fundo, borda, etc.)
        props = dict(boxstyle='round,pad=0.5', facecolor=COLORS["blush"], alpha=0.95, edgecolor=COLORS["prune"])
        
        # Adiciona a caixa de texto no canto superior esquerdo do gráfico
        ax.text(0.03, 0.97, legend_text, 
                transform=ax.transAxes, # Usa coordenadas relativas ao eixo (0 a 1)
                fontsize=9, 
                verticalalignment='top', 
                horizontalalignment='left', 
                bbox=props,
                color=COLORS["text_dark"])
        

        fig.tight_layout()
        canvas.draw()
        
        # Botão de insight (código existente)
        ctk.CTkButton(self.tab_group, text="💡 O que isso significa?", corner_radius=10, fg_color=COLORS["mauve"], text_color=COLORS["blush"], command=lambda: messagebox.showinfo("Insight sobre TBR", "Uma Razão Teta/Beta (TBR) mais alta no grupo TDAH é um biomarcador comum, sugerindo uma maturação cerebral mais lenta. A separação por gênero permite investigar se há diferenças neste padrão.")).pack(pady=10, padx=20)
        
    def create_individual_analysis_tab(self):
        """Cria a estrutura da aba de Análise Individual com todos os controles."""
        for widget in self.tab_individual.winfo_children(): widget.destroy()
        
        control_panel = ctk.CTkFrame(self.tab_individual, fg_color=COLORS["blush"])
        control_panel.pack(fill="x", pady=10, padx=10)

        ctk.CTkLabel(control_panel, text="Caso TDAH:", text_color=COLORS["text_dark"]).pack(side="left", padx=(10,5))
        self.adhd_tbr_choice_var = ctk.StringVar(value="TBR Mais Alto")
        ctk.CTkComboBox(control_panel, variable=self.adhd_tbr_choice_var, values=["TBR Mais Alto", "TBR Mais Baixo"]).pack(side="left", padx=5)
        
        ctk.CTkLabel(control_panel, text="Caso Controle:", text_color=COLORS["text_dark"]).pack(side="left", padx=(20,5))
        self.ctrl_tbr_choice_var = ctk.StringVar(value="TBR Mais Baixo")
        ctk.CTkComboBox(control_panel, variable=self.ctrl_tbr_choice_var, values=["TBR Mais Alto", "TBR Mais Baixo"]).pack(side="left", padx=5)

        ctk.CTkButton(control_panel, text="Atualizar Análise", command=self.update_individual_plots).pack(side="right", padx=10)
        
        self.wavelet_frame = ctk.CTkFrame(self.tab_individual, fg_color="transparent")
        self.wavelet_frame.pack(fill="both", expand=True, pady=5)
        self.spectrogram_frame = ctk.CTkFrame(self.tab_individual, fg_color="transparent")
        self.spectrogram_frame.pack(fill="both", expand=True, pady=5)

    def update_individual_plots(self):
        """Encontra os sujeitos selecionados e plota os gráficos lado a lado."""
        all_adhd_tbr = self.analysis_results['tbr_by_group']['F-ADHD'] + self.analysis_results['tbr_by_group']['M-ADHD']
        all_adhd_signals = self.loaded_data['F-ADHD'] + self.loaded_data['M-ADHD']
        all_ctrl_tbr = self.analysis_results['tbr_by_group']['F-Ctrl'] + self.analysis_results['tbr_by_group']['M-Ctrl']
        all_ctrl_signals = self.loaded_data['F-Ctrl'] + self.loaded_data['M-Ctrl']

        if not all_adhd_signals or not all_ctrl_signals: return

        idx_adhd = np.argmax(all_adhd_tbr) if self.adhd_tbr_choice_var.get() == "TBR Mais Alto" else np.argmin(all_adhd_tbr)
        idx_ctrl = np.argmax(all_ctrl_tbr) if self.ctrl_tbr_choice_var.get() == "TBR Mais Alto" else np.argmin(all_ctrl_tbr)
        
        signal_adhd = all_adhd_signals[idx_adhd][0, :]
        signal_ctrl = all_ctrl_signals[idx_ctrl][0, :]

        for widget in self.wavelet_frame.winfo_children(): widget.destroy()
        for widget in self.spectrogram_frame.winfo_children(): widget.destroy()

        fig_wav, (ax_wav_adhd, ax_wav_ctrl) = plt.subplots(1, 2, figsize=(10, 3.5))
        canvas_wav = FigureCanvasTkAgg(fig_wav, master=self.wavelet_frame)
        canvas_wav.get_tk_widget().pack(fill="both", expand=True)
        power_a, freqs_a, times_a = analyze_signal_with_wavelet(signal_adhd, FS)
        im_a = ax_wav_adhd.contourf(times_a, freqs_a, power_a, levels=20, cmap='viridis')
        self._apply_plot_style(ax_wav_adhd, fig_wav, f"Wavelet TDAH ({self.adhd_tbr_choice_var.get()})")
        fig_wav.colorbar(im_a, ax=ax_wav_adhd)
        power_c, freqs_c, times_c = analyze_signal_with_wavelet(signal_ctrl, FS)
        im_c = ax_wav_ctrl.contourf(times_c, freqs_c, power_c, levels=20, cmap='viridis')
        self._apply_plot_style(ax_wav_ctrl, fig_wav, f"Wavelet Controle ({self.ctrl_tbr_choice_var.get()})")
        fig_wav.colorbar(im_c, ax=ax_wav_ctrl)
        fig_wav.tight_layout()
        canvas_wav.draw()

        fig_spec, (ax_spec_adhd, ax_spec_ctrl) = plt.subplots(1, 2, figsize=(10, 3.5))
        canvas_spec = FigureCanvasTkAgg(fig_spec, master=self.spectrogram_frame)
        canvas_spec.get_tk_widget().pack(fill="both", expand=True)
        freqs_sa, times_sa, Sxx_a = calculate_spectrogram(signal_adhd, FS)
        im_sa = ax_spec_adhd.pcolormesh(times_sa, freqs_sa, 10*np.log10(Sxx_a), shading='gouraud', cmap='viridis')
        self._apply_plot_style(ax_spec_adhd, fig_spec, f"Espectrograma TDAH ({self.adhd_tbr_choice_var.get()})")
        fig_spec.colorbar(im_sa, ax=ax_spec_adhd)
        freqs_sc, times_sc, Sxx_c = calculate_spectrogram(signal_ctrl, FS)
        im_sc = ax_spec_ctrl.pcolormesh(times_sc, freqs_sc, 10*np.log10(Sxx_c), shading='gouraud', cmap='viridis')
        self._apply_plot_style(ax_spec_ctrl, fig_spec, f"Espectrograma Controle ({self.ctrl_tbr_choice_var.get()})")
        fig_spec.colorbar(im_sc, ax=ax_spec_ctrl)
        fig_spec.tight_layout()
        canvas_spec.draw()

    def create_average_analysis_tab(self):
        for widget in self.tab_average.winfo_children(): widget.destroy()
        
        self.gender_selector = ctk.CTkSegmentedButton(self.tab_average, values=["Geral", "Feminino", "Masculino"], command=self.plot_average_analyses, fg_color=COLORS["mauve"])
        self.gender_selector.set("Geral")
        self.gender_selector.pack(pady=10)
        self.average_plots_frame = ctk.CTkFrame(self.tab_average, fg_color="transparent")
        self.average_plots_frame.pack(fill="both", expand=True)
        
    def plot_average_analyses(self, mode="Geral"):
        for widget in self.average_plots_frame.winfo_children(): widget.destroy()

        if mode == "Geral":
            adhd_signals = self.loaded_data['F-ADHD'] + self.loaded_data['M-ADHD']
            ctrl_signals = self.loaded_data['F-Ctrl'] + self.loaded_data['M-Ctrl']
        elif mode == "Feminino":
            adhd_signals = self.loaded_data['F-ADHD']
            ctrl_signals = self.loaded_data['F-Ctrl']
        else: # Masculino
            adhd_signals = self.loaded_data['M-ADHD']
            ctrl_signals = self.loaded_data['M-Ctrl']
        
        fig_spec, (ax_spec_adhd, ax_spec_ctrl) = plt.subplots(1, 2, figsize=(10, 4))
        canvas_spec = FigureCanvasTkAgg(fig_spec, master=self.average_plots_frame)
        canvas_spec.get_tk_widget().pack(fill="x", pady=5)

        if adhd_signals:
            min_len = min(s.shape[1] for s in adhd_signals)
            all_sxx = [calculate_spectrogram(s[0, :min_len], FS)[2] for s in adhd_signals]
            avg_sxx = np.mean(all_sxx, axis=0)
            freqs, times, _ = calculate_spectrogram(adhd_signals[0][0, :min_len], FS)
            im = ax_spec_adhd.pcolormesh(times, freqs, 10 * np.log10(avg_sxx), shading='gouraud', cmap='viridis')
            self._apply_plot_style(ax_spec_adhd, fig_spec, f"Espectrograma Médio TDAH ({mode})")
            fig_spec.colorbar(im, ax=ax_spec_adhd)

        if ctrl_signals:
            min_len = min(s.shape[1] for s in ctrl_signals)
            all_sxx = [calculate_spectrogram(s[0, :min_len], FS)[2] for s in ctrl_signals]
            avg_sxx = np.mean(all_sxx, axis=0)
            freqs, times, _ = calculate_spectrogram(ctrl_signals[0][0, :min_len], FS)
            im = ax_spec_ctrl.pcolormesh(times, freqs, 10 * np.log10(avg_sxx), shading='gouraud', cmap='viridis')
            self._apply_plot_style(ax_spec_ctrl, fig_spec, f"Espectrograma Médio Controle ({mode})")
            fig_spec.colorbar(im, ax=ax_spec_ctrl)
        
        fig_spec.tight_layout()
        canvas_spec.draw()

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    app = EEGAnalyzerApp()
    app.mainloop()