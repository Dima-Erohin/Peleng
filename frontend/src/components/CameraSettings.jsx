import React from "react";
import { Select } from "./Dropdown";

export default function CameraSettings({
  cameraId,
  resolution,
  resolutions,
  fps,
  devices,
  selectedDevice,
  running,
  onCameraId,
  onResolution,
  onFps,
  onChangeCamera,
}) {
  const resolutionOptions = resolutions.map((size) => ({
    value: size,
    label: size.replace("x", " × "),
  }));
  const fpsOptions = [
    { value: "30", label: "30 к/с" },
    { value: "60", label: "60 к/с" },
  ];
  const deviceOptions = devices.map((device, index) => ({
    value: device.deviceId,
    label: device.label || `Камера ${index + 1}`,
  }));

  return (
    <>
      <label className="phone-field">
        <span>ID камеры</span>
        <input value={cameraId} onChange={(event) => onCameraId(event.target.value)} placeholder="например, phone1" disabled={running} />
      </label>
      <div className="phone-fields">
        <label className="phone-field">
          <span>Разрешение</span>
          <Select label="Разрешение" value={resolution} options={resolutionOptions} onChange={onResolution} disabled={running} />
        </label>
        <label className="phone-field">
          <span>Частота кадров</span>
          <Select label="Частота кадров" value={fps} options={fpsOptions} onChange={onFps} disabled={running} />
        </label>
      </div>
      {devices.length > 1 && (
        <label className="phone-field">
          <span>Источник видео</span>
          <Select
            label="Источник видео"
            value={selectedDevice}
            options={deviceOptions}
            onChange={onChangeCamera}
          />
        </label>
      )}
    </>
  );
}
