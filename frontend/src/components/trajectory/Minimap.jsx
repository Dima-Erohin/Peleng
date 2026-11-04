import React, { useEffect, useRef, useState } from "react";
import { Raycaster, Vector2 } from "three";
import { minimapStyle } from "../../trajectory/minimap-style.js";

export default function Minimap({ dataset, liveTrack, hiddenTracks, selectedId, time, onSelect }) {
  const containerRef = useRef(null);
  const engineRef = useRef(null);
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let observer;
    let cleanupPointer;
    import("../../trajectory/minimap-scene.js").then(({ MinimapScene }) => {
      if (cancelled) return;
      const engine = new MinimapScene(containerRef.current, setError);
      engineRef.current = engine;
      const canvas = engine.renderer.domElement;
      const raycaster = new Raycaster();
      let down = null;
      const pointerDown = (event) => { down = [event.clientX, event.clientY]; };
      const pointerUp = (event) => {
        if (!down || Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 5) { down = null; return; }
        down = null;
        const rect = canvas.getBoundingClientRect();
        raycaster.setFromCamera(new Vector2(
          (event.clientX - rect.left) / rect.width * 2 - 1,
          -(event.clientY - rect.top) / rect.height * 2 + 1,
        ), engine.camera);
        const objects = [...engine.objects.values()].filter((object) => object.enabled && object.group.visible);
        const hits = raycaster.intersectObjects(objects.map((object) => object.group), true);
        const hit = hits[0]?.object;
        if (!hit) return;
        const object = objects.find((item) => {
          for (let node = hit; node; node = node.parent) if (node === item.group) return true;
          return false;
        });
        if (object) selectRef.current(object.track.id);
      };
      canvas.addEventListener("pointerdown", pointerDown);
      canvas.addEventListener("pointerup", pointerUp);
      cleanupPointer = () => {
        canvas.removeEventListener("pointerdown", pointerDown);
        canvas.removeEventListener("pointerup", pointerUp);
      };
      observer = new IntersectionObserver(([entry]) => engine.setActive(entry.isIntersecting));
      observer.observe(containerRef.current);
      setReady(true);
    }).catch((reason) => { if (!cancelled) setError(`3D-мини-карта: ${reason.message}`); });
    return () => {
      cancelled = true;
      observer?.disconnect();
      cleanupPointer?.();
      engineRef.current?.dispose();
      engineRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    setError("");
    const engine = engineRef.current;
    engine.setDataset(dataset);
    engine.focus(true);
  }, [dataset, ready]);
  useEffect(() => { if (ready) engineRef.current.setLiveTrack(liveTrack); }, [liveTrack, dataset, ready]);
  useEffect(() => {
    if (!ready) return;
    const engine = engineRef.current;
    for (const id of engine.objects.keys()) engine.setObjectVisible(id, !hiddenTracks.includes(id));
    engine.select(selectedId);
    engine.seek(time);
  }, [time, selectedId, hiddenTracks, liveTrack, dataset, ready]);

  const tracks = [...dataset.tracks, ...(liveTrack ? [liveTrack] : [])].filter((track) => !hiddenTracks.includes(track.id));
  return (
    <section className="replay-minimap" aria-label="Мини-карта">
      <div className="replay-section-heading"><h2>Обзор маршрута</h2><span>3D · XYZ</span></div>
      <div className="minimap-viewport" ref={containerRef} />
      {error && <p role="alert">{error}</p>}
      <div className="minimap-actions">
        <button type="button" disabled={!ready} onClick={() => engineRef.current.focus(true)}>Общий вид</button>
      </div>
      <div className="minimap-legend">
        {tracks.map((track) => <button key={track.id} type="button" aria-pressed={track.id === selectedId}
          onClick={() => onSelect(track.id)}><i style={{
            background: minimapStyle(track).dashed
              ? `repeating-linear-gradient(to right, ${minimapStyle(track).color} 0 4px, transparent 4px 7px)`
              : minimapStyle(track).color,
            width: 18, height: 3, borderRadius: 0,
          }} />{track.source === "reference" ? "Эталон" : track.source === "estimate" || track.live ? "Оценка модели" : track.name}</button>)}
      </div>
      <p>Золотая сплошная — эталон · розовый пунктир — оценка. Бирюзовые метки — камеры.</p>
    </section>
  );
}
