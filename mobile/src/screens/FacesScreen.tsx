import React, { useCallback, useEffect, useState } from "react";
import {
  Alert,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import Animated from "react-native-reanimated";
import LoadingOverlay from "../components/LoadingOverlay";
import ActionButton from "../components/ActionButton";
import EmptyState from "../components/EmptyState";
import SectionLabel from "../components/SectionLabel";
import { enterPresets, useScalePress } from "../animations";
import { deleteIdentity, listIdentities, registerFace } from "../api/client";
import type { IdentityItem } from "../types/api";
import { colors, typography, spacing, radii, shadows } from "../theme";

const AVATAR_COLORS = [
  "#7c6cff",
  "#ff6b9d",
  "#00d9ff",
  "#34d399",
  "#fbbf24",
  "#f87171",
  "#a78bfa",
  "#38bdf8",
];

function getAvatarColor(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

export default function FacesScreen() {
  const [identities, setIdentities] = useState<IdentityItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("");
  const [registering, setRegistering] = useState(false);
  const [inputFocused, setInputFocused] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await listIdentities();
      setIdentities(res.items);
    } catch {
      // silently fail
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleRegister = async () => {
    if (!name.trim()) return Alert.alert("Enter a name");
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) return Alert.alert("Gallery permission required");

    const result = await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });
    if (result.canceled) return;

    setRegistering(true);
    try {
      const res = await registerFace(result.assets[0].uri, name.trim());
      Alert.alert("Registered", res.message);
      setName("");
      refresh();
    } catch (e: any) {
      Alert.alert("Failed", e.message);
    } finally {
      setRegistering(false);
    }
  };

  const handleDelete = (item: IdentityItem) => {
    Alert.alert("Delete", `Remove "${item.name}"?`, [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          setLoading(true);
          try {
            await deleteIdentity(item.identity_id);
            refresh();
          } catch (e: any) {
            Alert.alert("Failed", e.message);
          } finally {
            setLoading(false);
          }
        },
      },
    ]);
  };

  const renderItem = ({ item, index }: { item: IdentityItem; index: number }) => {
    const avatarColor = getAvatarColor(item.name);
    return (
      <Animated.View
        entering={enterPresets.stagger(index)}
        style={[styles.card, shadows.sm]}
      >
        <View style={[styles.avatar, { backgroundColor: avatarColor + "25" }]}>
          <Text style={[styles.avatarText, { color: avatarColor }]}>
            {item.name.charAt(0).toUpperCase()}
          </Text>
        </View>
        <View style={styles.cardInfo}>
          <Text style={styles.cardName}>{item.name}</Text>
          <Text style={styles.cardMeta}>
            {item.num_embeddings} embedding(s)
          </Text>
        </View>
        <Pressable
          style={styles.deleteBtn}
          onPress={() => handleDelete(item)}
        >
          <Ionicons name="trash-outline" size={18} color={colors.error} />
        </Pressable>
      </Animated.View>
    );
  };

  return (
    <View style={styles.container}>
      <LoadingOverlay visible={registering} message="Registering face..." />
      <LoadingOverlay visible={loading} message="Deleting..." />

      {/* Register Form */}
      <Animated.View
        entering={enterPresets.fadeInDown}
        style={[styles.form, shadows.sm]}
      >
        <SectionLabel text="Register New Face" />
        <View style={styles.formRow}>
          <TextInput
            style={[
              styles.input,
              inputFocused && styles.inputFocused,
            ]}
            placeholder="Enter name"
            placeholderTextColor={colors.textMuted}
            value={name}
            onChangeText={setName}
            onFocus={() => setInputFocused(true)}
            onBlur={() => setInputFocused(false)}
          />
          <View style={styles.registerBtnWrap}>
            <ActionButton
              title="Register"
              icon="person-add"
              onPress={handleRegister}
            />
          </View>
        </View>
      </Animated.View>

      {/* Identity List */}
      <FlatList
        data={identities}
        keyExtractor={(item) => item.identity_id}
        contentContainerStyle={styles.list}
        ListEmptyComponent={
          <EmptyState
            icon="people-outline"
            title="No Faces Registered"
            subtitle="Add a face above to get started with recognition"
          />
        }
        renderItem={renderItem}
        onRefresh={refresh}
        refreshing={false}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  form: {
    backgroundColor: colors.surface,
    padding: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  formRow: {
    flexDirection: "row",
    gap: spacing.sm,
    marginTop: spacing.sm,
  },
  input: {
    flex: 1,
    backgroundColor: colors.surfaceLight,
    borderRadius: radii.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    color: colors.text,
    ...typography.body,
    borderWidth: 1,
    borderColor: colors.border,
  },
  inputFocused: {
    borderColor: colors.primary,
  },
  registerBtnWrap: {
    justifyContent: "center",
  },
  list: {
    padding: spacing.lg,
    paddingBottom: 100,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: colors.border,
  },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: {
    fontSize: 18,
    fontWeight: "700",
  },
  cardInfo: {
    flex: 1,
    marginLeft: spacing.md,
  },
  cardName: {
    ...typography.h3,
    color: colors.text,
  },
  cardMeta: {
    ...typography.caption,
    color: colors.textMuted,
    marginTop: 2,
  },
  deleteBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.errorMuted,
    alignItems: "center",
    justifyContent: "center",
  },
});
