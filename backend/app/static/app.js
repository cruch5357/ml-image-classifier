function classifierApp() {
  return {
    selectedFile: null,
    previewUrl: "",
    predictions: [],
    loading: false,
    error: "",

    onPickFile(e) {
      const file = e.target.files?.[0];
      if (!file) return;
      this.setFile(file);
    },

    onDrop(e) {
      const file = e.dataTransfer.files?.[0];
      if (!file) return;
      this.setFile(file);
    },

    setFile(file) {
      if (!file.type.startsWith("image/")) {
        this.error = "Por favor sube un archivo de imagen.";
        return;
      }
      this.error = "";
      this.predictions = [];
      this.selectedFile = file;
      this.previewUrl = URL.createObjectURL(file);
    },

    async predict() {
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
