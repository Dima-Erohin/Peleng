import React from "react";
import { createRoot } from "react-dom/client";
import Viewer from "./viewer.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Viewer />
  </React.StrictMode>,
);
