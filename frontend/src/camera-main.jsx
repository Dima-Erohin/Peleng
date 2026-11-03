import React from "react";
import { createRoot } from "react-dom/client";
import Camera from "./camera.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Camera />
  </React.StrictMode>,
);
