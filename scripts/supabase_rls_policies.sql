-- Supabase RLS policies for the refactored Maize Disease Classifier app.
-- Safe with the current backend because server-side database calls use the service role.
-- Run this after the schema alignment script.

begin;

alter table public.users enable row level security;
alter table public.user_settings enable row level security;
alter table public.predictions enable row level security;
alter table public.system_logs enable row level security;
alter table public.feedback enable row level security;

alter table public.users force row level security;
alter table public.user_settings force row level security;
alter table public.predictions force row level security;
alter table public.system_logs force row level security;
alter table public.feedback force row level security;

drop policy if exists "users_select_own_or_admin" on public.users;
drop policy if exists "users_update_own_or_admin" on public.users;
drop policy if exists "users_insert_self_or_service" on public.users;

create policy "users_select_own_or_admin"
on public.users
for select
to authenticated
using (
    auth.uid() = id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

create policy "users_update_own_or_admin"
on public.users
for update
to authenticated
using (
    auth.uid() = id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
)
with check (
    auth.uid() = id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

create policy "users_insert_self_or_service"
on public.users
for insert
to authenticated
with check (auth.uid() = id);

drop policy if exists "user_settings_select_own_or_admin" on public.user_settings;
drop policy if exists "user_settings_insert_own_or_admin" on public.user_settings;
drop policy if exists "user_settings_update_own_or_admin" on public.user_settings;

create policy "user_settings_select_own_or_admin"
on public.user_settings
for select
to authenticated
using (
    auth.uid() = user_id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

create policy "user_settings_insert_own_or_admin"
on public.user_settings
for insert
to authenticated
with check (
    auth.uid() = user_id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

create policy "user_settings_update_own_or_admin"
on public.user_settings
for update
to authenticated
using (
    auth.uid() = user_id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
)
with check (
    auth.uid() = user_id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

drop policy if exists "predictions_select_own_or_admin" on public.predictions;
drop policy if exists "predictions_insert_own_or_admin" on public.predictions;
drop policy if exists "predictions_update_own_or_admin" on public.predictions;
drop policy if exists "predictions_delete_own_or_admin" on public.predictions;

create policy "predictions_select_own_or_admin"
on public.predictions
for select
to authenticated
using (
    auth.uid() = user_id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

create policy "predictions_insert_own_or_admin"
on public.predictions
for insert
to authenticated
with check (
    auth.uid() = user_id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

create policy "predictions_update_own_or_admin"
on public.predictions
for update
to authenticated
using (
    auth.uid() = user_id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
)
with check (
    auth.uid() = user_id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

create policy "predictions_delete_own_or_admin"
on public.predictions
for delete
to authenticated
using (
    auth.uid() = user_id
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

drop policy if exists "system_logs_select_admin_only" on public.system_logs;
drop policy if exists "system_logs_insert_admin_only" on public.system_logs;

create policy "system_logs_select_admin_only"
on public.system_logs
for select
to authenticated
using (
    exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

create policy "system_logs_insert_admin_only"
on public.system_logs
for insert
to authenticated
with check (
    exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

drop policy if exists "feedback_insert_public_or_authenticated" on public.feedback;
drop policy if exists "feedback_select_own_or_admin" on public.feedback;
drop policy if exists "feedback_update_admin_only" on public.feedback;

create policy "feedback_insert_public_or_authenticated"
on public.feedback
for insert
to anon, authenticated
with check (
    user_id is null or auth.uid() = user_id
);

create policy "feedback_select_own_or_admin"
on public.feedback
for select
to authenticated
using (
    user_id = auth.uid()
    or exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

create policy "feedback_update_admin_only"
on public.feedback
for update
to authenticated
using (
    exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
)
with check (
    exists (
        select 1
        from public.users as me
        where me.id = auth.uid()
        and me.is_admin = true
    )
);

commit;
