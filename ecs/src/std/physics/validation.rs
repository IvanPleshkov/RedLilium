use crate::{Entity, SystemError, Transform};

pub(super) fn invalid(entity: Entity, reason: &str) -> SystemError {
    SystemError::InvalidConfiguration {
        message: format!("physics entity {entity:?}: {reason}"),
    }
}

pub(super) fn transform(
    entity: Entity,
    t: &Transform,
    has_parent: bool,
) -> Result<(), SystemError> {
    if has_parent {
        return Err(invalid(
            entity,
            "rigid bodies must be roots (Parent is not supported)",
        ));
    }
    if t.scale != redlilium_core::math::Vec3::repeat(1.0) {
        return Err(invalid(
            entity,
            "rigid bodies require unit scale; put scaled visuals on a child",
        ));
    }
    if !t
        .translation
        .iter()
        .chain(t.rotation.coords.iter())
        .all(|x| x.is_finite())
        || !t.rotation.norm_squared().is_finite()
        || t.rotation.norm_squared() < 1e-12
    {
        return Err(invalid(
            entity,
            "pose must be finite with a nonzero quaternion",
        ));
    }
    Ok(())
}
