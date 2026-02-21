import React, { useCallback, useEffect, useState } from "react";
import {
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import Animated from "react-native-reanimated";
import GradientHeader from "../components/GradientHeader";
import PressableCard from "../components/PressableCard";
import StatusBadge from "../components/StatusBadge";
import { enterPresets } from "../animations";
import { checkHealth } from "../api/client";
import type { HealthResponse } from "../types/api";
import { API_BASE_URL } from "../constants";
import { colors, typography, spacing, radii, shadows } from "../theme";

export default function HomeScreen({ navigation }: any) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchHealth = useCallback(async () => {
    try {
      setError(null);
      const data = await checkHealth();
      setHealth(data);
    } catch (e: any) {
      setError(e.message || "Cannot connect to backend");
      setHealth(null);
    }
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchHealth();
    setRefreshing(false);
  };

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  const apiStatus = health?.status === "ok"
    ? "ok"
    : health?.status === "degraded"
    ? "degraded"
    : health
    ? "down"
    : error
    ? "down"
    : "checking";

  const statusLabel = health
    ? `API ${health.status.toUpperCase()}`
    : error
    ? "OFFLINE"
    : "Checking...";

  const navItems = [
    {
      screen: "Swap",
      icon: "swap-horizontal" as const,
      title: "Face Swap",
      desc: "Swap faces between two images",
      color: "#ff6b9d",
    },
    {
      screen: "Recognize",
      icon: "scan" as const,
      title: "Recognize",
      desc: "Detect and identify faces",
      color: colors.accent,
    },
    {
      screen: "Faces",
      icon: "people" as const,
      title: "Identities",
      desc: "Manage registered faces",
      color: colors.success,
    },
  ];

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={colors.primary}
        />
      }
    >
      <GradientHeader
        title="AI Face Recognition"
        subtitle="& Face Swap"
        icon="sparkles"
      />

      {/* Status Card */}
      <Animated.View
        entering={enterPresets.stagger(0)}
        style={[styles.statusCard, shadows.sm]}
      >
        <View style={styles.statusHeader}>
          <StatusBadge status={apiStatus} label={statusLabel} />
          {health && (
            <View style={styles.versionPill}>
              <Text style={styles.versionText}>
                v{health.version} | {health.environment}
              </Text>
            </View>
          )}
        </View>

        {health && (
          <View style={styles.components}>
            {Object.entries(health.components).map(([key, comp]) => (
              <View key={key} style={styles.compRow}>
                <Ionicons
                  name={
                    comp.status === "ok"
                      ? "checkmark-circle"
                      : "close-circle"
                  }
                  size={16}
                  color={comp.status === "ok" ? colors.success : colors.error}
                />
                <Text style={styles.compName}>{key}</Text>
                <Text style={styles.compStatus}>
                  {comp.loaded ? "loaded" : "not loaded"}
                </Text>
              </View>
            ))}
          </View>
        )}

        {error && <Text style={styles.error}>{error}</Text>}
        <Text style={styles.urlHint}>{API_BASE_URL}</Text>
      </Animated.View>

      {/* Navigation Cards */}
      {navItems.map((item, index) => (
        <PressableCard
          key={item.screen}
          onPress={() => navigation.navigate(item.screen)}
          entering={enterPresets.stagger(index + 1)}
          style={styles.navCard}
        >
          <View style={styles.navRow}>
            <View
              style={[
                styles.iconCircle,
                { backgroundColor: item.color + "18" },
              ]}
            >
              <Ionicons name={item.icon} size={24} color={item.color} />
            </View>
            <View style={styles.navText}>
              <Text style={styles.navTitle}>{item.title}</Text>
              <Text style={styles.navDesc}>{item.desc}</Text>
            </View>
            <Ionicons
              name="chevron-forward"
              size={20}
              color={colors.textMuted}
            />
          </View>
        </PressableCard>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { paddingBottom: 100 },
  statusCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.lg,
    marginHorizontal: spacing.xl,
    marginTop: -spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  statusHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: spacing.md,
  },
  versionPill: {
    backgroundColor: colors.surfaceLight,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
  },
  versionText: {
    ...typography.caption,
    color: colors.textMuted,
  },
  components: {
    marginTop: spacing.sm,
    gap: spacing.sm,
  },
  compRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
  },
  compName: {
    ...typography.bodySmall,
    color: colors.textSecondary,
    flex: 1,
  },
  compStatus: {
    ...typography.caption,
    color: colors.textMuted,
  },
  error: {
    ...typography.bodySmall,
    color: colors.error,
    marginTop: spacing.sm,
  },
  urlHint: {
    ...typography.caption,
    color: colors.textMuted,
    marginTop: spacing.md,
  },
  navCard: {
    marginHorizontal: spacing.xl,
    marginTop: spacing.md,
  },
  navRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  iconCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: "center",
    justifyContent: "center",
  },
  navText: {
    flex: 1,
    marginLeft: spacing.md,
  },
  navTitle: {
    ...typography.h3,
    color: colors.text,
  },
  navDesc: {
    ...typography.bodySmall,
    color: colors.textSecondary,
    marginTop: 2,
  },
});
