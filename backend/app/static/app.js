function classifierApp() {
  return {
    verdict: null,
    demoChoice: "dog.png",
    selectedFile: null,
    previewUrl: "",
    predictions: [], // siempre [{label, prob}]
    loading: false,
    error: "",

    // ---------- helpers ----------
    _normalizePredictions(raw) {
      // raw puede ser: [[label, prob], ...] o [{label, prob}, ...]
      if (!Array.isArray(raw)) return [];

      // caso 1: [[label, prob], ...]
      if (Array.isArray(raw[0])) {
        return raw
          .map((p) => {
            const label = String(p?.[0] ?? "");
            const prob = Number(p?.[1]);
            return { label, prob: Number.isFinite(prob) ? prob : 0 };
          })
          .filter((p) => p.label);
      }

      // caso 2: [{label, prob}, ...]
      return raw
        .map((p) => {
          const label = String(p?.label ?? p?.class ?? "");
          const prob = Number(p?.prob ?? p?.score);
          return { label, prob: Number.isFinite(prob) ? prob : 0 };
        })
        .filter((p) => p.label);
    },

    _setPreview(file) {
      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = URL.createObjectURL(file);
    },

    // ---------- file handling ----------
    pick(e) {
      const file = e.target.files?.[0];
      if (!file) return;
      this.setFile(file);
      e.target.value = ""; // permite elegir el mismo archivo 2 veces
    },

    drop(e) {
      const file = e.dataTransfer.files?.[0];
      if (!file) return;
      this.setFile(file);
    },

    setFile(file) {
      if (!file.type || !file.type.startsWith("image/")) {
        this.error = "Por favor sube una imagen (PNG/JPG/WEBP).";
        return;
      }
      this.error = "";
      this.predictions = [];
      this.selectedFile = file;
      this._setPreview(file);
    },

    clear() {
      this.selectedFile = null;
      this.predictions = [];
      this.error = "";
      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = "";
      const inp = document.getElementById("fileInput");
      if (inp) inp.value = "";
      this.verdict = null;
      this.demoChoice = "";
    },

    // ---------- demo mode ----------
    async loadSample(filename) {
      this.error = "";
      this.predictions = [];
      this.loading = true;

      try {
        const url = `/static/samples/${filename}`;
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) throw new Error("No se pudo cargar el sample.");

        const blob = await res.blob();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });

        this.setFile(file);
      } catch (e) {
        this.error = "No se pudo cargar el ejemplo.";
      } finally {
        this.loading = false;
      }
    },

    async demoRun() {
      // carga sample seleccionado + predice
      await this.loadSample(this.sample);
      if (this.selectedFile) await this.predict();
    },

    // ---------- prediction ----------
    async predict() {
      if (!this.selectedFile) return;

      this.loading = true;
      this.error = "";
      this.predictions = [];

      try {
        const fd = new FormData();
        fd.append("file", this.selectedFile);

        const res = await fetch("/api/predict", { method: "POST", body: fd });
        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
          this.error = data?.error || "Error al predecir.";
          return;
        }

        const norm = this._normalizePredictions(data.predictions);
        if (!norm.length) {
          this.error = "Respuesta válida, pero sin predicciones.";
          return;
        }

        // ordena desc
        this.predictions = data.predictions || [];
        this.verdict = data.verdict || null;
      } catch (err) {
        this.error = "No se pudo conectar con el servidor.";
      } finally {
        this.loading = false;
      }
    },
  };
}
