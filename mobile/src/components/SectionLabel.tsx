import React from "react";
import { StyleSheet, Text } from "react-native";
import { colors, typography, spacing } from "../theme";

interface Props {
  text: string;
}

export default function SectionLabel({ text }: Props) {
  return <Text style={styles.label}>{text}</Text>;
}

const styles = StyleSheet.create({
  label: {
    ...typography.label,
    color: colors.textMuted,
    marginBottom: spacing.sm,
  },
});
