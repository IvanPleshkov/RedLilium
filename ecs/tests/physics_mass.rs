//! Final mass properties, density weighting and transactional descriptor edits.
#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_core::math::{Vec3, quat_from_rotation_z};
use redlilium_ecs::*;

struct RemoveCollider(Entity);
impl System for RemoveCollider {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        let e = self.0;
        ctx.commands(move |w| {
            w.despawn(e);
        });
        Ok(())
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-5 * b.abs().max(1.0), "{a} != {b}");
}
macro_rules! tests {
    () => {
        fn world()->World {
            let mut w=World::new(); register_std_components(&mut w); w.insert_resource(Physics::default());
            w.resource_mut::<Physics>().gravity=Default::default(); w
        }
        fn sync(w:&mut World,regular:bool)->Vec<SystemError> {
            let mut s=SystemsContainer::new();
            if regular {s.add(SyncRegular);} else {s.add_exclusive(Sync);}
            EcsRunner::multi_thread(2).run(w,&s)
        }
        fn step(w:&mut World)->Vec<SystemError> {
            let mut s=SystemsContainer::new();s.add(Step);EcsRunner::multi_thread(2).run(w,&s)
        }
        fn valid(w:&mut World,regular:bool) {let errors=sync(w,regular);assert!(errors.is_empty(),"{errors:?}");}
        fn invalid(errors:Vec<SystemError>) {
            assert!(matches!(errors.as_slice(),[SystemError::InvalidConfiguration{message}] if message.contains("mass_properties")),"{errors:?}");
        }
        fn fixture(w:&mut World)->(Entity,Entity,Entity) {
            let b=w.spawn_with((Transform::IDENTITY,Body::dynamic())).unwrap();
            let a=w.spawn_with((Owner{body:b},Collider::ball(0.5).with_density(1.0).with_local_pose(pose(-1.0)))).unwrap();
            let c=w.spawn_with((Owner{body:b},Collider::ball(0.5).with_density(3.0).with_local_pose(pose(3.0)))).unwrap();
            (b,a,c)
        }
        fn props(w:&World,b:Entity)->NativeMass {
            let p=w.resource::<Physics>();let h=p.body_for_entity(b).unwrap();p.bodies()[h].mass_properties().local_mprops
        }
        fn full(mass:f32)->Settings {
            Settings::new(mass).with_center_of_mass(Some(vec(0.2,-0.3))).with_inertia(Some(inertia(2.0)))
        }
        #[test]
        fn final_mass_scales_central_inertia_and_preserves_density_weighted_center() {
            for regular in [false,true] {
                let mut w=world();let(b,a,c)=fixture(&mut w);valid(&mut w,regular);
                let base=props(&w,b);close(base.local_com.x as f64,2.0);
                let handle=w.resource::<Physics>().body_for_entity(b).unwrap();
                let ah=w.resource::<Physics>().collider_for_entity(a).unwrap();
                w.get_mut::<Body>(b).unwrap().mass_properties=Some(Settings::new((base.mass()*2.0) as f32));valid(&mut w,regular);
                let doubled=props(&w,b);close(doubled.mass() as f64,base.mass() as f64*2.0);close(doubled.local_com.x as f64,2.0);
                for (x,y) in moments(doubled).into_iter().zip(moments(base)) {close(x,y*2.0);}
                assert_eq!(w.resource::<Physics>().body_for_entity(b),Some(handle));
                assert_eq!(w.resource::<Physics>().collider_for_entity(a),Some(ah));
                for e in [a,c] {let p=w.resource::<Physics>();let h=p.collider_for_entity(e).unwrap();close(p.colliders()[h].density() as f64,0.0);}
                assert!(step(&mut w).is_empty());close(props(&w,b).mass() as f64,doubled.mass() as f64);
                w.get_mut::<Body>(b).unwrap().mass_properties.as_mut().unwrap().center_of_mass=Some(vec(-4.0,-2.0));valid(&mut w,regular);
                let shifted=props(&w,b);close(shifted.local_com.x as f64,-4.0);
                for (x,y) in moments(shifted).into_iter().zip(moments(doubled)) {close(x,y);}
                assert_eq!(w.resource::<Physics>().bodies()[handle].translation(),NativeVector::ZERO);
                w.get_mut::<Body>(b).unwrap().mass_properties=None;valid(&mut w,regular);
                let restored=props(&w,b);close(restored.mass() as f64,base.mass() as f64);close(restored.local_com.x as f64,2.0);
                for (x,y) in moments(restored).into_iter().zip(moments(base)) {close(x,y);}
                for (e,density) in [(a,1.0),(c,3.0)] {let p=w.resource::<Physics>();let h=p.collider_for_entity(e).unwrap();close(p.colliders()[h].density() as f64,density);}
            }
        }
        #[test]
        fn explicit_body_without_colliders_has_correct_impulse_response_and_keeps_velocities_on_edit() {
            for regular in [false,true] {
                let mut w=world();let b=w.spawn_with((Transform::IDENTITY,Body::dynamic().with_mass_properties(Some(full(10.0))))).unwrap();valid(&mut w,regular);
                let h=w.resource::<Physics>().body_for_entity(b).unwrap();
                w.resource_mut::<Physics>().body_motion(h).unwrap().apply_impulse(native_vec(20.0,0.0),true);
                close(w.resource::<Physics>().bodies()[h].linvel().x as f64,2.0);
                let old_pose=*w.resource::<Physics>().bodies()[h].position();
                w.get_mut::<Body>(b).unwrap().mass_properties=Some(full(5.0));valid(&mut w,regular);
                assert_eq!(*w.resource::<Physics>().bodies()[h].position(),old_pose);
                close(w.resource::<Physics>().bodies()[h].linvel().x as f64,2.0);
                assert!(step(&mut w).is_empty());close(props(&w,b).mass() as f64,5.0);
            }
        }
        #[test]
        fn collider_edits_recompute_only_derived_properties_and_reparenting_updates_both_bodies() {
            for regular in [false,true] {
                let mut w=world();let(b,a,c)=fixture(&mut w);w.get_mut::<Body>(b).unwrap().mass_properties=Some(Settings::new(12.0));valid(&mut w,regular);
                let bh=w.resource::<Physics>().body_for_entity(b).unwrap();let ch=w.resource::<Physics>().collider_for_entity(c).unwrap();
                w.get_mut::<Collider>(c).unwrap().density=1.0;valid(&mut w,regular);close(props(&w,b).local_com.x as f64,1.0);
                w.get_mut::<Collider>(c).unwrap().local_pose=pose(5.0);valid(&mut w,regular);close(props(&w,b).local_com.x as f64,2.0);
                // The source becomes compound without changing its collider identity.
                w.get_mut::<Collider>(c).unwrap().shape=Collider::compound(vec![Part::ball(0.5),Part::ball(0.5)]).shape;
                valid(&mut w,regular);close(props(&w,b).local_com.x as f64,3.0);close(props(&w,b).mass() as f64,12.0);
                assert_eq!(w.resource::<Physics>().collider_for_entity(c),Some(ch));
                let other=w.spawn_with((Transform::IDENTITY,Body::dynamic())).unwrap();
                w.get_mut::<Owner>(c).unwrap().body=other;valid(&mut w,regular);
                close(props(&w,b).local_com.x as f64,-1.0);close(props(&w,b).mass() as f64,12.0);
                close(props(&w,other).local_com.x as f64,5.0);assert!(props(&w,other).mass()>0.0);
                // Removing the last source is rejected before changing existing physics.
                w.remove::<Collider>(a).unwrap();invalid(sync(&mut w,regular));
                assert_eq!(w.resource::<Physics>().body_for_entity(b),Some(bh));
                assert!(w.resource::<Physics>().collider_for_entity(a).is_some());
                w.get_mut::<Body>(b).unwrap().mass_properties=Some(full(12.0));valid(&mut w,regular);
                assert!(w.resource::<Physics>().collider_for_entity(a).is_none());close(props(&w,b).mass() as f64,12.0);
            }
        }
        #[test]
        fn unrelated_edits_do_not_replace_mass_cache_and_noop_preserves_sleep() {
            for regular in [false,true] {
                let mut w=world();let(b,a,_)=fixture(&mut w);w.get_mut::<Body>(b).unwrap().mass_properties=Some(Settings::new(7.0));valid(&mut w,regular);
                let h=w.resource::<Physics>().body_for_entity(b).unwrap();
                let ptr=w.resource::<Physics>().bodies()[h].mass_properties().additional_local_mprops.as_deref().unwrap() as *const _;
                let mass=props(&w,b);
                w.resource_mut::<Physics>().body_motion(h).unwrap().sleep();valid(&mut w,regular);
                assert!(w.resource::<Physics>().bodies()[h].is_sleeping());
                w.get_mut::<Collider>(a).unwrap().friction=0.9;valid(&mut w,regular);
                assert_eq!(w.resource::<Physics>().bodies()[h].mass_properties().additional_local_mprops.as_deref().unwrap() as *const _,ptr);
                assert_eq!(props(&w,b),mass);
                assert!(step(&mut w).is_empty());close(props(&w,b).mass() as f64,7.0);
                w.get_mut::<Body>(b).unwrap().linear_damping=0.4;valid(&mut w,regular);
                assert_eq!(w.resource::<Physics>().bodies()[h].mass_properties().additional_local_mprops.as_deref().unwrap() as *const _,ptr);
            }
        }
        #[test]
        fn invalid_authoring_or_missing_mass_sources_leave_the_entire_batch_unchanged() {
            for regular in [false,true] {
                let mut w=world();let(b,_,_)=fixture(&mut w);valid(&mut w,regular);let base=props(&w,b);
                let mut bad=vec![Settings::new(0.0),Settings::new(-1.0),Settings::new(f32::NAN),Settings::new(f32::INFINITY),full(1.0).with_center_of_mass(Some(vec(f32::NAN,0.0))),full(1.0).with_inertia(Some(inertia(0.0))),full(1.0).with_inertia(Some(inertia(-1.0))),full(1.0).with_inertia(Some(inertia(f32::INFINITY)))];
                extra_invalid(&mut bad);
                for settings in bad {
                    let new=w.spawn_with((Transform::IDENTITY,Body::dynamic())).unwrap();
                    w.get_mut::<Body>(b).unwrap().mass_properties=Some(settings);invalid(sync(&mut w,regular));
                    assert_eq!(props(&w,b),base);assert!(w.resource::<Physics>().body_for_entity(new).is_none());w.despawn(new);
                }
                w.get_mut::<Body>(b).unwrap().mass_properties=None;valid(&mut w,regular);
                let empty=w.spawn_with((Transform::IDENTITY,Body::dynamic().with_mass_properties(Some(Settings::new(1.0))))).unwrap();
                invalid(sync(&mut w,regular));
                w.insert(empty,Collider::ball(1.0).with_density(0.0)).unwrap();invalid(sync(&mut w,regular));
                w.get_mut::<Body>(empty).unwrap().mass_properties=Some(Settings::new(1.0).with_center_of_mass(Some(vec(0.0,0.0))));invalid(sync(&mut w,regular));
                w.get_mut::<Body>(empty).unwrap().mass_properties=Some(Settings::new(1.0).with_inertia(Some(inertia(1.0))));invalid(sync(&mut w,regular));
                w.get_mut::<Body>(empty).unwrap().mass_properties=Some(full(1.0));valid(&mut w,regular);
            }
        }
        #[test]
        fn deferred_cancellation_recomputes_survivors_or_blocks_step_until_mass_is_valid() {
            for remove_all in [false,true] {
                let mut w=world();let(b,a,c)=fixture(&mut w);
                if remove_all {w.despawn(a);}
                w.get_mut::<Body>(b).unwrap().mass_properties=Some(Settings::new(6.0));
                let mut s=SystemsContainer::new();s.add(RemoveCollider(c));s.add(SyncRegular);s.add_edge::<RemoveCollider,SyncRegular>().unwrap();
                assert!(EcsRunner::multi_thread(2).run(&mut w,&s).is_empty());
                if remove_all {
                    invalid(step(&mut w));invalid(sync(&mut w,true));
                    w.get_mut::<Body>(b).unwrap().mass_properties=Some(full(6.0));valid(&mut w,true);
                } else {close(props(&w,b).local_com.x as f64,-1.0);}
                assert!(step(&mut w).is_empty());close(props(&w,b).mass() as f64,6.0);
            }
        }
        #[test]
        fn settings_roundtrip_through_scene_formats_and_apply_to_all_body_types() {
            let mut w=world();
            let settings=[None,Some(Settings::new(5.0)),Some(Settings::new(6.0).with_center_of_mass(Some(vec(0.0,-0.4)))),Some(full(8.0))];
            for (i,(kind,settings)) in [Body::dynamic(),Body::fixed(),Body::kinematic_position(),Body::kinematic_velocity()].into_iter().zip(settings).enumerate() {
                w.spawn_with((Name::new(format!("mass{i}")),Transform::IDENTITY,kind.with_mass_properties(settings),Collider::ball(0.5))).unwrap();
            }
            let snapshot=w.serialize_world().unwrap();let mut snapshots=vec![snapshot.clone()];
            #[cfg(feature="serialize-ron")]
            snapshots.push(serialize::decode(&serialize::encode(&snapshot,serialize::Format::Ron).unwrap(),serialize::Format::Ron).unwrap());
            #[cfg(feature="serialize-bincode")]
            snapshots.push(serialize::decode(&serialize::encode(&snapshot,serialize::Format::Bincode).unwrap(),serialize::Format::Bincode).unwrap());
            for snapshot in snapshots {
                let mut target=world();let entities=target.deserialize_world_into(&snapshot).unwrap();valid(&mut target,false);
                for (i,s) in settings.into_iter().enumerate() {
                    let e=*entities.iter().find(|e|target.get::<Name>(**e).unwrap().0==format!("mass{i}")).unwrap();
                    assert_eq!(target.get::<Body>(e).unwrap().mass_properties,s);
                    if let Some(s)=s {close(props(&target,e).mass() as f64,s.mass as f64);}
                    target.get_mut::<Body>(e).unwrap().body_type=Body::dynamic().body_type;
                }
                valid(&mut target,true);assert!(step(&mut target).is_empty());
            }
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod dim2 {
    use super::*;
    use physics::MassSettings2D as Settings;
    use physics::components2d::{
        Collider2D as Collider, ColliderBody2D as Owner, ColliderPart2D as Part,
        ColliderPose2D as Local, RigidBody2D as Body,
    };
    use physics::rapier2d::prelude::{MassProperties as NativeMass, Vector as NativeVector};
    use physics::systems2d::{
        StepPhysics2D as Step, SyncPhysicsBodies2D as Sync,
        SyncPhysicsBodiesSystem2D as SyncRegular,
    };
    use physics::world2d::PhysicsWorld2D as Physics;
    fn vec(x: f32, y: f32) -> redlilium_core::math::Vec2 {
        redlilium_core::math::Vec2::new(x, y)
    }
    fn native_vec(x: f32, y: f32) -> NativeVector {
        NativeVector::new(x as _, y as _)
    }
    fn pose(x: f32) -> Local {
        Local {
            translation: vec(x, 0.0),
            ..Default::default()
        }
    }
    fn inertia(v: f32) -> f32 {
        v
    }
    fn moments(m: NativeMass) -> Vec<f64> {
        vec![m.principal_inertia() as f64]
    }
    fn extra_invalid(_: &mut Vec<Settings>) {}
    tests!();
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod dim3 {
    use super::*;
    use physics::MassSettings3D as Settings;
    use physics::components3d::{
        Collider3D as Collider, ColliderBody3D as Owner, ColliderPart3D as Part,
        ColliderPose3D as Local, RigidBody3D as Body,
    };
    use physics::rapier3d::prelude::{MassProperties as NativeMass, Vector as NativeVector};
    use physics::systems3d::{
        StepPhysics3D as Step, SyncPhysicsBodies3D as Sync,
        SyncPhysicsBodiesSystem3D as SyncRegular,
    };
    use physics::world3d::PhysicsWorld3D as Physics;
    fn vec(x: f32, y: f32) -> redlilium_core::math::Vec3 {
        redlilium_core::math::Vec3::new(x, y, 0.0)
    }
    fn native_vec(x: f32, y: f32) -> NativeVector {
        NativeVector::new(x as _, y as _, 0.0)
    }
    fn pose(x: f32) -> Local {
        Local {
            translation: vec(x, 0.0),
            ..Default::default()
        }
    }
    fn inertia(v: f32) -> physics::AngularInertia3D {
        physics::AngularInertia3D::diagonal(Vec3::repeat(v))
    }
    fn moments(m: NativeMass) -> Vec<f64> {
        m.principal_inertia()
            .to_array()
            .into_iter()
            .map(|v| v as f64)
            .collect()
    }
    fn extra_invalid(out: &mut Vec<Settings>) {
        out.push(
            full(1.0).with_inertia(Some(physics::AngularInertia3D::diagonal(Vec3::new(
                1.0, 1.0, 3.0,
            )))),
        );
        out.push(full(1.0).with_inertia(Some(physics::AngularInertia3D {
            principal: Vec3::repeat(1.0),
            rotation: redlilium_core::math::Quat::new(0.0, 0.0, 0.0, 0.0),
        })));
    }
    #[test]
    fn rotated_principal_axes_control_torque_and_survive_center_override() {
        let mut w = world();
        let inertia = physics::AngularInertia3D {
            principal: Vec3::new(2.0, 3.0, 4.0),
            rotation: quat_from_rotation_z(::std::f32::consts::FRAC_PI_2),
        };
        let b = w
            .spawn_with((
                Transform::IDENTITY,
                Body::dynamic().with_mass_properties(Some(full(10.0).with_inertia(Some(inertia)))),
            ))
            .unwrap();
        valid(&mut w, false);
        let h = w.resource::<Physics>().body_for_entity(b).unwrap();
        w.resource_mut::<Physics>()
            .body_motion(h)
            .unwrap()
            .apply_torque_impulse(NativeVector::new(3.0, 0.0, 0.0), true);
        close(w.resource::<Physics>().bodies()[h].angvel().x as f64, 1.0);
        close(w.resource::<Physics>().bodies()[h].angvel().y as f64, 0.0);
        w.get_mut::<Body>(b)
            .unwrap()
            .mass_properties
            .as_mut()
            .unwrap()
            .center_of_mass = Some(Vec3::new(0.0, -1.0, 0.0));
        valid(&mut w, true);
        close(w.resource::<Physics>().bodies()[h].angvel().x as f64, 1.0);
    }
    tests!();
}
