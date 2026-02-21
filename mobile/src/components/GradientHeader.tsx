import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { colors, typography, spacing, radii } from "../theme";

interface Props {
  title: string;
  subtitle?: string;
  icon?: keyof typeof Ionicons.glyphMap;
  compact?: boolean;
}

export default function GradientHeader({ title, subtitle, icon, compact }: Props) {
  return (
    <LinearGradient
      colors={[colors.primary, colors.primaryDark, colors.background]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
      style={[styles.gradient, compact && styles.compact]}
    >
      <View style={styles.content}>
        {icon && (
          <View style={styles.iconWrap}>
            <Ionicons name={icon} size={compact ? 24 : 36} color={colors.text} />
          </View>
        )}
        <Text style={[styles.title, compact && styles.titleCompact]}>{title}</Text>
        {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: {
    paddingTop: 52,
    paddingBottom: spacing.xxl,
    paddingHorizontal: spacing.xl,
    borderBottomLeftRadius: radii.xl,
    borderBottomRightRadius: radii.xl,
  },
  compact: {
    paddingTop: 48,
    paddingBottom: spacing.lg,
  },
  content: {
    alignItems: "center",
  },
  iconWrap: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "rgba(255,255,255,0.15)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.md,
  },
  title: {
    ...typography.hero,
    color: colors.text,
    textAlign: "center",
  },
  titleCompact: {
    ...typography.h1,
  },
  subtitle: {
    ...typography.body,
    color: "rgba(255,255,255,0.7)",
    textAlign: "center",
    marginTop: spacing.xs,
  },
});
