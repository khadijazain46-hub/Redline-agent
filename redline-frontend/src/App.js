import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [aiEnabled, setAiEnabled] = useState(true);
  const [downloadLink, setDownloadLink] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setDownloadLink("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please select a DXF file");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("ai", aiEnabled);

    try {
      const resp = await axios.post(
        "http://127.0.0.1:8000/api/upload",
        formData,
        { responseType: "blob" } // important for file download
      );

      // Create downloadable link
      const url = window.URL.createObjectURL(new Blob([resp.data]));
      setDownloadLink(url);
    } catch (err) {
      console.error(err);
      alert("Upload failed");
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Redline Agent - AI CAD Reviewer</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".dxf,.dwg" onChange={handleFileChange} />
        <div>
          <label>
            <input
              type="checkbox"
              checked={aiEnabled}
              onChange={() => setAiEnabled(!aiEnabled)}
            />{" "}
            AI Review
          </label>
        </div>
        <button type="submit">Upload & Redline</button>
      </form>

      {downloadLink && (
        <div style={{ marginTop: "1rem" }}>
          <a href={downloadLink} download="output_redlined.dxf">
            Download Redlined DXF
          </a>
        </div>
      )}
    </div>
  );
}

export default App;
