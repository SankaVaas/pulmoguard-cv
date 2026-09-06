import { useRef, useState, type DragEvent } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/bmp"];

export function UploadDropzone({ onFileSelected, disabled }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFile(file: File | undefined) {
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      alert(`Unsupported file type: ${file.type}. Please upload a JPEG, PNG, or BMP image.`);
      return;
    }
    setPreviewUrl(URL.createObjectURL(file));
    onFileSelected(file);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    handleFile(e.dataTransfer.files?.[0]);
  }

  return (
    <div
      className={`dropzone ${isDragging ? "dropzone--active" : ""} ${disabled ? "dropzone--disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files?.[0])}
        disabled={disabled}
      />
      {previewUrl ? (
        <img src={previewUrl} alt="X-ray preview" className="dropzone-preview" />
      ) : (
        <div className="dropzone-placeholder">
          <p>Drag & drop a chest X-ray image here</p>
          <p className="dropzone-subtext">or click to browse (JPEG, PNG, BMP)</p>
        </div>
      )}
    </div>
  );
}
