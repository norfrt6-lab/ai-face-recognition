import { TextStyle, ViewStyle } from "react-native";

export const colors = {
  background: "#0a0a14",
  surface: "#14142a",
  surfaceLight: "#1e1e3a",
  primary: "#7c6cff",
  primaryDark: "#5a4fcc",
  primaryMuted: "rgba(124, 108, 255, 0.12)",
  accent: "#00d9ff",
  success: "#34d399",
  successMuted: "rgba(52, 211, 153, 0.10)",
  warning: "#fbbf24",
  error: "#f87171",
  errorMuted: "rgba(248, 113, 113, 0.10)",
  text: "#f0f0f5",
  textSecondary: "#9999b0",
  textMuted: "#555570",
  border: "#2a2a45",
  borderLight: "#3a3a55",
  overlay: "rgba(0, 0, 0, 0.75)",
};

export const typography: Record<string, TextStyle> = {
  hero: { fontSize: 32, fontWeight: "800", letterSpacing: -0.5 },
  h1: { fontSize: 24, fontWeight: "700", letterSpacing: -0.3 },
  h2: { fontSize: 20, fontWeight: "600" },
  h3: { fontSize: 16, fontWeight: "600" },
  body: { fontSize: 15, fontWeight: "400" },
  bodySmall: { fontSize: 13, fontWeight: "400" },
  caption: { fontSize: 11, fontWeight: "400" },
  label: {
    fontSize: 11,
    fontWeight: "600",
    letterSpacing: 1.2,
    textTransform: "uppercase",
  },
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 28,
  xxxl: 40,
};

export const radii = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 999,
};

export const shadows: Record<string, ViewStyle> = {
  sm: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 3,
  },
  md: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
};

export function glowShadow(color: string): ViewStyle {
  return {
    shadowColor: color,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 8,
  };
}
