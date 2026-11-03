import React, { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Select } from "./Dropdown";

function getLabel(select) {
  if (select.getAttribute("aria-label")) return select.getAttribute("aria-label");
  if (select.id === "accScale") return "Масштаб кольца точности";
  if (select.id === "objMode") return "Режим обнаружения объектов";
  if (select.id === "tjSel") return "Выбор траектории";
  const label = select.closest("label");
  const text = label
    ? [...label.childNodes]
      .filter((node) => node.nodeType === Node.TEXT_NODE)
      .map((node) => node.textContent.trim())
      .filter(Boolean)
      .join(" ")
    : "";
  return text || select.title || select.name || "Выбор";
}

function ExtensionSelect({ element }) {
  const [snapshot, setSnapshot] = useState(() => readSelect(element));
  const refresh = useCallback(() => setSnapshot(readSelect(element)), [element]);

  useEffect(() => {
    element.addEventListener("change", refresh);
    const observer = new MutationObserver(refresh);
    observer.observe(element, { childList: true, subtree: true, attributes: true, characterData: true });
    return () => {
      element.removeEventListener("change", refresh);
      observer.disconnect();
    };
  }, [element, refresh]);

  return (
    <Select
      label={getLabel(element)}
      value={snapshot.value}
      options={snapshot.options}
      onChange={(value) => {
        element.value = value;
        element.dispatchEvent(new Event("change", { bubbles: true }));
        refresh();
      }}
      disabled={element.disabled}
    />
  );
}

function readSelect(element) {
  return {
    value: element.value,
    options: [...element.options].map((option) => ({
      value: option.value,
      label: option.textContent.trim(),
    })),
  };
}

export default function ExtensionDropdowns() {
  const [selects, setSelects] = useState([]);

  useEffect(() => {
    const root = document.getElementById("ext");
    if (!root) return undefined;
    const hosts = new Map();
    const scan = () => {
      for (const select of root.querySelectorAll("select")) {
        if (hosts.has(select)) continue;
        const host = document.createElement("span");
        host.className = "peleng-select-host";
        select.classList.add("peleng-select-source");
        select.insertAdjacentElement("afterend", host);
        hosts.set(select, host);
      }

      for (const [select, host] of hosts) {
        if (select.isConnected && root.contains(select)) continue;
        host.remove();
        hosts.delete(select);
      }
      setSelects([...hosts.entries()]);
    };

    scan();
    const observer = new MutationObserver(scan);
    observer.observe(root, { childList: true, subtree: true });
    return () => {
      observer.disconnect();
      for (const [select, host] of hosts) {
        select.classList.remove("peleng-select-source");
        host.remove();
      }
    };
  }, []);

  return selects.map(([element, host]) => (
    createPortal(<ExtensionSelect element={element} />, host, element.id || undefined)
  ));
}
