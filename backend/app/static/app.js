function classifierApp() {
  return {
    selectedFile: null,
    previewUrl: "",
    predictions: [],
    loading: false,
    error: "",

    pick(e) {
      const file = e.target.files?.[0];
      if (!file) return;
      this.setFile(file);
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

      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = URL.createObjectURL(file);
    },

    clear() {
      this.selectedFile = null;
      this.predictions = [];
      this.error = "";
      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = "";
    },

    async predict() {
      if (!this.selectedFile) return;

      this.loading = true;
      this.error = "";
      this.predictions = [];

      try {
        const fd = new FormData();
        fd.append("file", this.selectedFile);

        const res = await fetch("/api/predict", { method: "POST", body: fd });
        const data = await res.json();

        if (!res.ok) {
          this.error = data?.error || "Error al predecir.";
        } else {
          this.predictions = data.predictions || [];
        }
      } catch (err) {
        this.error = "No se pudo conectar con el servidor.";
      } finally {
        this.loading = false;
      }
    },
  };
}
