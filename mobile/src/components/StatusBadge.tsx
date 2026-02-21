import React from "react";
import { StyleSheet, Text, View } from "react-native";
import Animated from "react-native-reanimated";
import { usePulse } from "../animations";
import { colors, typography, spacing, radii } from "../theme";

type Status = "ok" | "degraded" | "down" | "checking";

interface Props {
  status: Status;
  label: string;
}

const statusColors: Record<Status, string> = {
  ok: colors.success,
  degraded: colors.warning,
  down: colors.error,
  checking: colors.textMuted,
};

export default function StatusBadge({ status, label }: Props) {
  const pulseStyle = usePulse(status === "ok");
  const color = statusColors[status];

  return (
    <View style={[styles.badge, { borderColor: color + "30" }]}>
      <Animated.View
        style={[styles.dot, { backgroundColor: color }, pulseStyle]}
      />
      <Text style={[styles.label, { color }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radii.full,
    borderWidth: 1,
    backgroundColor: "rgba(0,0,0,0.2)",
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: spacing.sm,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: "600",
  },
});
